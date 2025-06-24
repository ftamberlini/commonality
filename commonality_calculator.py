import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from typing import List, Dict, Tuple

class CommonalityCalculator:
    def __init__(self):
        """Inicializa o calculador de commonality"""
        pass
    
    def calculate_browsing_probability(self, position: int, gamma: float) -> float:
        """
        Calcula a probabilidade de navegação usando Rank-biased Precision
        Pr(k) = (1-γ)γ^(k-1)
        """
        return (1 - gamma) * (gamma ** (position - 1))
    
    def calculate_recall(self, ranking: List[Dict], k: int, category_items: List[int], 
                        total_category_items: int) -> float:
        """
        Calcula o recall para uma categoria específica
        R(π,k,g) = |π₁:ₖ ∩ Dₘ| / |Dₘ|
        """
        if total_category_items == 0:
            return 0
            
        items_up_to_k = ranking[:k]
        category_items_in_top_k = sum(1 for item in items_up_to_k 
                                    if item['id'] in category_items)
        
        return category_items_in_top_k / total_category_items
    
    def calculate_expected_familiarity(self, ranking: List[Dict], category_items: List[int],
                                     total_category_items: int, gamma: float) -> float:
        """
        Calcula a familiaridade esperada para um usuário
        Pr(F_{u,g}|π_u) = Σ Pr(i)R(π_u,i,g)
        """
        expected_familiarity = 0
        
        for k in range(1, len(ranking) + 1):
            prob = self.calculate_browsing_probability(k, gamma)
            recall = self.calculate_recall(ranking, k, category_items, total_category_items)
            expected_familiarity += prob * recall
            
        return expected_familiarity
    
    def calculate_category_commonality(self, familiarities: List[float]) -> float:
        """
        Calcula a commonality para uma categoria
        C_g(π) = ∏ Pr(F_{u,g}|π_u)
        """
        if not familiarities:
            return 0
        return np.prod(familiarities)
    
    def generate_simulated_data(self, num_items: int, categories: List[Dict]) -> List[Dict]:
        """Gera catálogo de itens simulado"""
        catalog = []
        item_index = 0
        
        for cat_idx, category in enumerate(categories):
            items_in_category = int(num_items * category['size'])
            for _ in range(items_in_category):
                catalog.append({
                    'id': item_index,
                    'category': category['name'],
                    'category_index': cat_idx
                })
                item_index += 1
        
        # Completar com itens sem categoria específica
        while len(catalog) < num_items:
            catalog.append({
                'id': item_index,
                'category': 'Outros',
                'category_index': -1
            })
            item_index += 1
            
        return catalog
    
    def generate_ranking(self, catalog: List[Dict], diversity_bias: float) -> List[Dict]:
        """Gera ranking simulado baseado no viés do algoritmo"""
        ranked_catalog = []
        
        for item in catalog:
            # Score base aleatório
            base_score = random.random() * (1 - diversity_bias)
            # Bonus para itens de categorias específicas
            category_bonus = diversity_bias if item['category_index'] >= 0 else 0
            # Ruído adicional
            noise = random.random() * 0.1
            
            ranked_catalog.append({
                **item,
                'score': base_score + category_bonus + noise
            })
        
        # Ordenar por score decrescente
        return sorted(ranked_catalog, key=lambda x: x['score'], reverse=True)
    
    def simulate_recommender_systems(self, num_users: int, num_items: int, 
                                   gamma: float, categories: List[Dict]) -> List[Dict]:
        """Simula diferentes sistemas de recomendação"""
        catalog = self.generate_simulated_data(num_items, categories)
        results = []
        
        # Diferentes estratégias de recomendação
        algorithms = [
            {'name': 'Personalizado', 'bias': 0.1},
            {'name': 'Popularidade', 'bias': 0.3},
            {'name': 'Diversificado', 'bias': 0.7},
            {'name': 'Balanceado', 'bias': 0.5}
        ]
        
        for algorithm in algorithms:
            user_familiarities = []
            category_results = {cat['name']: [] for cat in categories}
            
            # Para cada usuário
            for user_id in range(num_users):
                user_familiarity = {}
                
                # Gerar ranking simulado
                ranking = self.generate_ranking(catalog, algorithm['bias'])
                
                # Calcular familiaridade para cada categoria
                for category in categories:
                    category_items = [item['id'] for item in catalog 
                                    if item['category'] == category['name']]
                    
                    familiarity = self.calculate_expected_familiarity(
                        ranking, category_items, len(category_items), gamma
                    )
                    
                    user_familiarity[category['name']] = familiarity
                    category_results[category['name']].append(familiarity)
                
                user_familiarities.append(user_familiarity)
            
            # Calcular commonality por categoria
            category_commonalities = {}
            total_commonality = 1
            
            for category in categories:
                fam_values = category_results[category['name']]
                commonality = self.calculate_category_commonality(fam_values)
                category_commonalities[category['name']] = commonality
                total_commonality *= commonality
            
            # Média geométrica
            geometric_mean_commonality = total_commonality ** (1/len(categories)) if categories else 0
            
            # Familiaridade média
            avg_familiarity = np.mean([
                np.mean(category_results[cat['name']]) for cat in categories
            ]) if categories else 0
            
            results.append({
                'algorithm': algorithm['name'],
                'user_familiarities': user_familiarities,
                'category_commonalities': category_commonalities,
                'total_commonality': geometric_mean_commonality,
                'average_familiarity': avg_familiarity
            })
        
        return results

def main():
    st.set_page_config(
        page_title="Simulação da Métrica Commonality",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Simulação da Métrica Commonality")
    st.markdown("""
    **Implementação da métrica proposta por Ferraro et al. para avaliar sistemas de recomendação 
    na promoção de diversidade cultural compartilhada.**
    """)
    
    # Sidebar com controles
    st.sidebar.header("Parâmetros da Simulação")
    
    num_users = st.sidebar.slider("Número de Usuários", 3, 20, 5)
    num_items = st.sidebar.slider("Itens no Catálogo", 10, 100, 20)
    gamma = st.sidebar.slider("Parâmetro de Paciência (γ)", 0.1, 0.95, 0.8, 0.05)
    
    st.sidebar.markdown("**Controla quão profundamente os usuários navegam nas listas**")
    
    # Configuração de categorias
    st.sidebar.header("Categorias Editoriais")
    
    # Categorias padrão
    default_categories = [
        {'name': 'Diretoras Mulheres', 'size': 0.15, 'color': '#FF6B6B'},
        {'name': 'Cinema Independente', 'size': 0.25, 'color': '#4ECDC4'},
        {'name': 'Cinema Não-Ocidental', 'size': 0.20, 'color': '#45B7D1'}
    ]
    
    categories = []
    for i, cat in enumerate(default_categories):
        st.sidebar.subheader(f"Categoria {i+1}")
        name = st.sidebar.text_input(f"Nome da Categoria {i+1}", cat['name'])
        size = st.sidebar.slider(f"Proporção no catálogo {i+1}", 0.05, 0.5, cat['size'], 0.05)
        categories.append({'name': name, 'size': size, 'color': cat['color']})
    
    # Fórmulas matemáticas
    st.header("Fórmulas Implementadas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Probabilidade de Navegação:**")
        st.latex(r"Pr(k) = (1-\gamma)\gamma^{k-1}")
        
        st.markdown("**Familiaridade Esperada:**")
        st.latex(r"Pr(F_{u,g}|\pi_u) = \sum Pr(i)R(\pi_u,i,g)")
    
    with col2:
        st.markdown("**Recall de Categoria:**")
        st.latex(r"R(\pi,k,g) = \frac{|\pi_{1:k} \cap D_g|}{|D_g|}")
        
        st.markdown("**Commonality:**")
        st.latex(r"C_g(\pi) = \prod Pr(F_{u,g}|\pi_u)")
    
    # Executar simulação
    calculator = CommonalityCalculator()
    
    with st.spinner("Executando simulação..."):
        results = calculator.simulate_recommender_systems(num_users, num_items, gamma, categories)
    
    # Preparar dados para visualização
    chart_data = []
    for result in results:
        row = {
            'Algoritmo': result['algorithm'],
            'Commonality Total': result['total_commonality'],
            'Familiaridade Média': result['average_familiarity']
        }
        
        # Adicionar commonality por categoria
        for cat in categories:
            row[f"{cat['name']}_commonality"] = result['category_commonalities'][cat['name']]
        
        chart_data.append(row)
    
    df_results = pd.DataFrame(chart_data)
    
    # Visualizações
    st.header("Resultados da Simulação")
    
    # Gráfico de Commonality Total
    st.subheader("Commonality por Algoritmo")
    fig_total = px.bar(
        df_results, 
        x='Algoritmo', 
        y='Commonality Total',
        title="Commonality Total por Algoritmo",
        color='Commonality Total',
        color_continuous_scale='viridis'
    )
    fig_total.update_layout(height=400)
    st.plotly_chart(fig_total, use_container_width=True)
    
    # Gráfico de Commonality por Categoria
    st.subheader("Commonality por Categoria")
    
    category_data = []
    for _, row in df_results.iterrows():
        for cat in categories:
            category_data.append({
                'Algoritmo': row['Algoritmo'],
                'Categoria': cat['name'],
                'Commonality': row[f"{cat['name']}_commonality"],
                'Cor': cat['color']
            })
    
    df_categories = pd.DataFrame(category_data)
    
    fig_cat = px.bar(
        df_categories,
        x='Algoritmo',
        y='Commonality',
        color='Categoria',
        title="Commonality por Categoria e Algoritmo",
        barmode='group'
    )
    fig_cat.update_layout(height=400)
    st.plotly_chart(fig_cat, use_container_width=True)
    
    # Scatter plot: Trade-off
    st.subheader("Trade-off: Commonality vs Familiaridade Média")
    
    fig_scatter = px.scatter(
        df_results,
        x='Familiaridade Média',
        y='Commonality Total',
        color='Algoritmo',
        size='Commonality Total',
        title="Relação entre Commonality e Familiaridade Média",
        hover_data=['Algoritmo']
    )
    fig_scatter.update_layout(height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Análise detalhada
    st.header("🔍 Análise Detalhada")
    
    col1, col2 = st.columns(2)
    
    for i, result in enumerate(results):
        with col1 if i % 2 == 0 else col2:
            st.subheader(f"🤖 {result['algorithm']}")
            
            metrics_data = {
                'Métrica': ['Commonality Total', 'Familiaridade Média'] + 
                          [f"Commonality - {cat['name']}" for cat in categories],
                'Valor': [
                    f"{result['total_commonality']:.6f}",
                    f"{result['average_familiarity']:.4f}"
                ] + [
                    f"{result['category_commonalities'][cat['name']]:.6f}" 
                    for cat in categories
                ]
            }
            
            st.table(pd.DataFrame(metrics_data))
    
    # Interpretação
    st.header("Interpretação dos Resultados")
    
    st.info("""
    **Commonality:** Mede a probabilidade de que TODOS os usuários simultaneamente 
    se familiarizem com as categorias selecionadas. Valores próximos a 0 indicam que pelo menos 
    um usuário tem baixa familiaridade.
    
    **Trade-off:** Algoritmos muito personalizados podem ter alta familiaridade média 
    mas baixa commonality, pois diferentes usuários se familiarizam com diferentes categorias.
    
    **Parâmetro γ:** Valores altos (próximos a 1) simulam usuários que navegam mais 
    profundamente nas listas. Valores baixos simulam usuários que veem apenas os primeiros itens.
    """)
    
    # Dados para download
    st.header("Download dos Dados")
    
    csv = df_results.to_csv(index=False)
    st.download_button(
        label="📊 Baixar resultados em CSV",
        data=csv,
        file_name='commonality_results.csv',
        mime='text/csv'
    )

if __name__ == "__main__":
    main()
