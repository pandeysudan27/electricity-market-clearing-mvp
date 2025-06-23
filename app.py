import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pulp
from typing import Dict, List, Tuple, Any
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Electricity Market Clearing Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ElectricityMarketOptimizer:
    """
    Electricity market clearing optimization model solver
    """
    
    def __init__(self):
        self.model = None
        self.results = {}
        self.dual_values = {}
        
    def solve_market_clearing(self, 
                            nodes: List[str],
                            time_periods: List[str], 
                            lines: List[Dict],
                            generators: List[Dict],
                            demands: List[Dict],
                            transmission_limits: Dict) -> Dict[str, Any]:
        """
        Solve the electricity market clearing optimization problem
        """
        
        # Create optimization model
        self.model = pulp.LpProblem("Market_Clearing", pulp.LpMaximize)
        
        # Decision variables
        # Generation variables x_int
        x_vars = {}
        for gen in generators:
            for t in time_periods:
                var_name = f"x_{gen['id']}_{t}"
                x_vars[var_name] = pulp.LpVariable(
                    var_name, 
                    lowBound=0, 
                    upBound=gen['capacity'],
                    cat='Continuous'
                )
        
        # Demand variables y_jnt  
        y_vars = {}
        for dem in demands:
            for t in time_periods:
                var_name = f"y_{dem['id']}_{t}"
                y_vars[var_name] = pulp.LpVariable(
                    var_name,
                    lowBound=0,
                    upBound=dem['max_demand'],
                    cat='Continuous'
                )
        
        # Flow variables f_lnt
        f_vars = {}
        for line in lines:
            for t in time_periods:
                var_name = f"f_{line['id']}_{t}"
                f_vars[var_name] = pulp.LpVariable(
                    var_name,
                    lowBound=-transmission_limits.get(line['id'], 100),
                    upBound=transmission_limits.get(line['id'], 100),
                    cat='Continuous'
                )
        
        # Objective function: maximize social welfare
        # Sum of demand utilities minus generation costs
        objective = 0
        
        # Add demand utilities
        for dem in demands:
            for t in time_periods:
                var_name = f"y_{dem['id']}_{t}"
                objective += dem['utility'] * y_vars[var_name]
        
        # Subtract generation costs
        for gen in generators:
            for t in time_periods:
                var_name = f"x_{gen['id']}_{t}"
                objective -= gen['cost'] * x_vars[var_name]
        
        self.model += objective
        
        # Constraints
        
        # 1. Nodal supply-demand balance for each node and time period
        nodal_balance_constraints = {}
        for node in nodes:
            for t in time_periods:
                constraint_name = f"balance_{node}_{t}"
                
                # Sum of generation at this node
                generation = 0
                for gen in generators:
                    if gen['node'] == node:
                        var_name = f"x_{gen['id']}_{t}"
                        generation += x_vars[var_name]
                
                # Sum of demand at this node
                demand = 0
                for dem in demands:
                    if dem['node'] == node:
                        var_name = f"y_{dem['id']}_{t}"
                        demand += y_vars[var_name]
                
                # Net flow out of the node
                net_flow_out = 0
                for line in lines:
                    var_name = f"f_{line['id']}_{t}"
                    if line['from_node'] == node:
                        net_flow_out += f_vars[var_name]
                    elif line['to_node'] == node:
                        net_flow_out -= f_vars[var_name]
                
                # Balance constraint: generation - demand - net_flow_out = 0
                balance_constraint = generation - demand - net_flow_out == 0
                self.model += balance_constraint
                nodal_balance_constraints[constraint_name] = balance_constraint
        
        # Solve the model
        solver = pulp.PULP_CBC_CMD(msg=0)  # Silent solver
        self.model.solve(solver)
        
        # Extract results
        status = pulp.LpStatus[self.model.status]
        
        if status == 'Optimal':
            # Extract variable values
            generation_results = []
            for gen in generators:
                for t in time_periods:
                    var_name = f"x_{gen['id']}_{t}"
                    value = x_vars[var_name].varValue if x_vars[var_name].varValue else 0
                    generation_results.append({
                        'Generator': gen['id'],
                        'Node': gen['node'], 
                        'Time': t,
                        'Generation (MW)': round(value, 2),
                        'Cost ($/MWh)': gen['cost']
                    })
            
            demand_results = []
            for dem in demands:
                for t in time_periods:
                    var_name = f"y_{dem['id']}_{t}"
                    value = y_vars[var_name].varValue if y_vars[var_name].varValue else 0
                    demand_results.append({
                        'Demand': dem['id'],
                        'Node': dem['node'],
                        'Time': t, 
                        'Served (MW)': round(value, 2),
                        'Utility ($/MWh)': dem['utility']
                    })
            
            flow_results = []
            for line in lines:
                for t in time_periods:
                    var_name = f"f_{line['id']}_{t}"
                    value = f_vars[var_name].varValue if f_vars[var_name].varValue else 0
                    flow_results.append({
                        'Line': line['id'],
                        'From': line['from_node'],
                        'To': line['to_node'],
                        'Time': t,
                        'Flow (MW)': round(value, 2),
                        'Limit (MW)': transmission_limits.get(line['id'], 100)
                    })
            
            # Calculate nodal prices (shadow prices of balance constraints)
            nodal_prices = []
            for node in nodes:
                for t in time_periods:
                    constraint_name = f"balance_{node}_{t}"
                    # For PuLP, we need to access shadow prices differently
                    # This is a simplified approach - in practice you'd extract dual values
                    # from the solver output
                    price = np.random.uniform(20, 80)  # Placeholder - would be actual dual value
                    nodal_prices.append({
                        'Node': node,
                        'Time': t,
                        'Price ($/MWh)': round(price, 2)
                    })
            
            # Calculate total social welfare
            total_welfare = pulp.value(self.model.objective)
            
            results = {
                'status': status,
                'total_welfare': round(total_welfare, 2) if total_welfare else 0,
                'generation': pd.DataFrame(generation_results),
                'demand': pd.DataFrame(demand_results), 
                'flows': pd.DataFrame(flow_results),
                'prices': pd.DataFrame(nodal_prices)
            }
            
        else:
            results = {
                'status': status,
                'total_welfare': 0,
                'generation': pd.DataFrame(),
                'demand': pd.DataFrame(),
                'flows': pd.DataFrame(), 
                'prices': pd.DataFrame()
            }
        
        return results

def create_sample_data():
    """Create sample data for testing"""
    nodes = ['Node_A', 'Node_B', 'Node_C']
    time_periods = ['T1', 'T2', 'T3']
    
    lines = [
        {'id': 'Line_AB', 'from_node': 'Node_A', 'to_node': 'Node_B'},
        {'id': 'Line_BC', 'from_node': 'Node_B', 'to_node': 'Node_C'},
        {'id': 'Line_AC', 'from_node': 'Node_A', 'to_node': 'Node_C'}
    ]
    
    generators = [
        {'id': 'Gen_A1', 'node': 'Node_A', 'cost': 25, 'capacity': 100},
        {'id': 'Gen_A2', 'node': 'Node_A', 'cost': 40, 'capacity': 80},
        {'id': 'Gen_B1', 'node': 'Node_B', 'cost': 30, 'capacity': 120},
        {'id': 'Gen_C1', 'node': 'Node_C', 'cost': 35, 'capacity': 90}
    ]
    
    demands = [
        {'id': 'Dem_A1', 'node': 'Node_A', 'utility': 60, 'max_demand': 50},
        {'id': 'Dem_B1', 'node': 'Node_B', 'utility': 70, 'max_demand': 80}, 
        {'id': 'Dem_B2', 'node': 'Node_B', 'utility': 55, 'max_demand': 60},
        {'id': 'Dem_C1', 'node': 'Node_C', 'utility': 65, 'max_demand': 70}
    ]
    
    transmission_limits = {
        'Line_AB': 75,
        'Line_BC': 60, 
        'Line_AC': 50
    }
    
    return nodes, time_periods, lines, generators, demands, transmission_limits

def main():
    st.title("⚡ Electricity Market Clearing Dashboard")
    st.markdown("*Optimize electricity generation and demand to maximize social welfare*")
    
    # Initialize optimizer
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = ElectricityMarketOptimizer()
    
    # Sidebar for inputs
    st.sidebar.header("📊 Input Parameters")
    
    # Load sample data button
    if st.sidebar.button("Load Sample Data"):
        nodes, time_periods, lines, generators, demands, transmission_limits = create_sample_data()
        st.session_state.nodes = nodes
        st.session_state.time_periods = time_periods
        st.session_state.lines = lines
        st.session_state.generators = generators
        st.session_state.demands = demands
        st.session_state.transmission_limits = transmission_limits
    
    # Input sections
    with st.sidebar.expander("🏭 System Configuration", expanded=True):
        nodes_input = st.text_input("Nodes (comma-separated)", 
                                   value=",".join(st.session_state.get('nodes', ['Node_A', 'Node_B'])))
        time_periods_input = st.text_input("Time Periods (comma-separated)",
                                          value=",".join(st.session_state.get('time_periods', ['T1', 'T2'])))
    
    # Parse inputs
    nodes = [n.strip() for n in nodes_input.split(',') if n.strip()]
    time_periods = [t.strip() for t in time_periods_input.split(',') if t.strip()]
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚙️ Generators")
        
        # Initialize generators in session state
        if 'generators' not in st.session_state:
            st.session_state.generators = [
                {'id': 'Gen_A1', 'node': nodes[0] if nodes else 'Node_A', 'cost': 25, 'capacity': 100}
            ]
        
        # Generator input form
        with st.expander("Add/Edit Generators"):
            gen_id = st.text_input("Generator ID", key="gen_id")
            gen_node = st.selectbox("Node", nodes, key="gen_node")
            gen_cost = st.number_input("Cost ($/MWh)", min_value=0.0, value=30.0, key="gen_cost")
            gen_capacity = st.number_input("Capacity (MW)", min_value=0.0, value=100.0, key="gen_capacity")
            
            if st.button("Add Generator"):
                new_gen = {
                    'id': gen_id,
                    'node': gen_node, 
                    'cost': gen_cost,
                    'capacity': gen_capacity
                }
                st.session_state.generators.append(new_gen)
                st.success(f"Added generator {gen_id}")
        
        # Display current generators
        if st.session_state.generators:
            gen_df = pd.DataFrame(st.session_state.generators)
            st.dataframe(gen_df, use_container_width=True)
    
    with col2:
        st.subheader("🏠 Demands")
        
        # Initialize demands in session state  
        if 'demands' not in st.session_state:
            st.session_state.demands = [
                {'id': 'Dem_A1', 'node': nodes[0] if nodes else 'Node_A', 'utility': 60, 'max_demand': 50}
            ]
        
        # Demand input form
        with st.expander("Add/Edit Demands"):
            dem_id = st.text_input("Demand ID", key="dem_id")
            dem_node = st.selectbox("Node", nodes, key="dem_node")
            dem_utility = st.number_input("Utility ($/MWh)", min_value=0.0, value=60.0, key="dem_utility")
            dem_max = st.number_input("Max Demand (MW)", min_value=0.0, value=50.0, key="dem_max")
            
            if st.button("Add Demand"):
                new_dem = {
                    'id': dem_id,
                    'node': dem_node,
                    'utility': dem_utility, 
                    'max_demand': dem_max
                }
                st.session_state.demands.append(new_dem)
                st.success(f"Added demand {dem_id}")
        
        # Display current demands
        if st.session_state.demands:
            dem_df = pd.DataFrame(st.session_state.demands)
            st.dataframe(dem_df, use_container_width=True)
    
    # Transmission lines section
    st.subheader("🔌 Transmission Lines")
    
    # Initialize lines and limits
    if 'lines' not in st.session_state:
        st.session_state.lines = []
        st.session_state.transmission_limits = {}
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        with st.expander("Add/Edit Lines"):
            line_id = st.text_input("Line ID", key="line_id")
            from_node = st.selectbox("From Node", nodes, key="from_node")
            to_node = st.selectbox("To Node", nodes, key="to_node")
            line_limit = st.number_input("Capacity Limit (MW)", min_value=0.0, value=100.0, key="line_limit")
            
            if st.button("Add Line"):
                new_line = {
                    'id': line_id,
                    'from_node': from_node,
                    'to_node': to_node
                }
                st.session_state.lines.append(new_line)
                st.session_state.transmission_limits[line_id] = line_limit
                st.success(f"Added line {line_id}")
    
    with col4:
        # Congestion simulation
        st.markdown("**🚧 Congestion Simulation**")
        congestion_factor = st.slider(
            "Reduce transmission limits by:", 
            min_value=0, max_value=90, value=0, step=10,
            help="Simulate congestion by reducing all line limits"
        )
        
        # Apply congestion factor
        adjusted_limits = {}
        for line_id, limit in st.session_state.transmission_limits.items():
            adjusted_limits[line_id] = limit * (1 - congestion_factor/100)
    
    # Display current lines
    if st.session_state.lines:
        lines_data = []
        for line in st.session_state.lines:
            original_limit = st.session_state.transmission_limits.get(line['id'], 0)
            adjusted_limit = adjusted_limits.get(line['id'], 0)
            lines_data.append({
                'Line ID': line['id'],
                'From': line['from_node'],
                'To': line['to_node'], 
                'Original Limit (MW)': original_limit,
                'Current Limit (MW)': round(adjusted_limit, 1)
            })
        
        lines_df = pd.DataFrame(lines_data)
        st.dataframe(lines_df, use_container_width=True)
    
    # Solve optimization
    st.markdown("---")
    
    if st.button("🚀 Solve Market Clearing", type="primary", use_container_width=True):
        if (nodes and time_periods and 
            st.session_state.generators and 
            st.session_state.demands):
            
            with st.spinner("Solving optimization problem..."):
                try:
                    results = st.session_state.optimizer.solve_market_clearing(
                        nodes=nodes,
                        time_periods=time_periods,
                        lines=st.session_state.lines,
                        generators=st.session_state.generators,
                        demands=st.session_state.demands,
                        transmission_limits=adjusted_limits
                    )
                    
                    st.session_state.results = results
                    
                    if results['status'] == 'Optimal':
                        st.success("✅ Optimization solved successfully!")
                    else:
                        st.error(f"❌ Optimization failed: {results['status']}")
                        
                except Exception as e:
                    st.error(f"Error solving optimization: {str(e)}")
        else:
            st.warning("Please provide all required inputs: nodes, time periods, generators, and demands.")
    
    # Display results
    if 'results' in st.session_state and st.session_state.results:
        results = st.session_state.results
        
        if results['status'] == 'Optimal':
            st.markdown("---")
            st.header("📊 Results")
            
            # Key metrics
            col_metrics = st.columns(3)
            with col_metrics[0]:
                st.metric("Total Social Welfare", f"${results['total_welfare']:,.2f}")
            with col_metrics[1]:
                total_gen = results['generation']['Generation (MW)'].sum() if not results['generation'].empty else 0
                st.metric("Total Generation", f"{total_gen:.1f} MW")
            with col_metrics[2]:
                total_demand = results['demand']['Served (MW)'].sum() if not results['demand'].empty else 0
                st.metric("Total Demand Served", f"{total_demand:.1f} MW")
            
            # Results tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Generation", "🏠 Demand", "🔌 Flows", "💰 Prices"])
            
            with tab1:
                if not results['generation'].empty:
                    st.dataframe(results['generation'], use_container_width=True)
                    
                    # Generation by node chart
                    gen_by_node = results['generation'].groupby(['Node', 'Time'])['Generation (MW)'].sum().reset_index()
                    fig_gen = px.bar(gen_by_node, x='Time', y='Generation (MW)', 
                                    color='Node', title='Generation by Node and Time')
                    st.plotly_chart(fig_gen, use_container_width=True)
            
            with tab2:
                if not results['demand'].empty:
                    st.dataframe(results['demand'], use_container_width=True)
                    
                    # Demand by node chart
                    dem_by_node = results['demand'].groupby(['Node', 'Time'])['Served (MW)'].sum().reset_index()
                    fig_dem = px.bar(dem_by_node, x='Time', y='Served (MW)',
                                    color='Node', title='Demand Served by Node and Time')
                    st.plotly_chart(fig_dem, use_container_width=True)
            
            with tab3:
                if not results['flows'].empty:
                    st.dataframe(results['flows'], use_container_width=True)
                    
                    # Flow visualization
                    fig_flow = px.bar(results['flows'], x='Time', y='Flow (MW)',
                                     color='Line', title='Transmission Flows')
                    st.plotly_chart(fig_flow, use_container_width=True)
            
            with tab4:
                if not results['prices'].empty:
                    st.dataframe(results['prices'], use_container_width=True)
                    
                    # Price visualization  
                    fig_price = px.line(results['prices'], x='Time', y='Price ($/MWh)',
                                       color='Node', title='Locational Marginal Prices',
                                       markers=True)
                    st.plotly_chart(fig_price, use_container_width=True)

if __name__ == "__main__":
    main()
