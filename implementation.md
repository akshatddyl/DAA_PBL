# Algorithmic Trading Strategy Engine - Implementation Guide

## Phase 1: Foundation & Setup (Week 1-2)

### Technology Stack

**Backend:**
- **Python 3.10+** (main language)
- **NumPy/Pandas** (data manipulation)
- **NetworkX** (graph algorithms)
- **SciPy** (optimization)
- **TA-Lib** or **pandas-ta** (technical indicators)

**Data:**
- **yfinance** or **Alpha Vantage API** (free historical data)
- **CSV files** (for offline testing)

**Visualization:**
- **Matplotlib/Seaborn** (static plots)
- **Plotly** (interactive graphs)
- **Dash** or **Streamlit** (web dashboard)

**Optional Advanced:**
- **PyTorch/TensorFlow** (if adding ML components)
- **Redis** (caching for real-time data)

### Project Structure

```
trading_engine/
├── data/
│   ├── ingestion.py          # Data fetching and preprocessing
│   ├── storage.py             # Data persistence
│   └── cache.py               # Data caching
├── models/
│   ├── market_graph.py        # Graph-based market model
│   ├── state_space.py         # State space representation
│   └── indicators.py          # Technical indicators
├── algorithms/
│   ├── dynamic_programming/
│   │   ├── optimal_trading.py # DP for buy/sell timing
│   │   └── portfolio_dp.py    # DP for portfolio allocation
│   ├── greedy/
│   │   ├── threshold_strategy.py
│   │   └── momentum_strategy.py
│   ├── graph/
│   │   ├── dijkstra_trading.py
│   │   ├── bellman_ford.py
│   │   └── astar_strategy.py
│   ├── optimization/
│   │   ├── branch_bound.py    # Branch & Bound
│   │   ├── knapsack.py        # Portfolio as knapsack
│   │   └── genetic.py         # Genetic algorithms
│   └── game_theory/
│       ├── minimax.py         # Minimax algorithm
│       ├── nash_equilibrium.py
│       └── evolutionary.py    # Evolutionary game theory
├── strategies/
│   ├── base_strategy.py       # Abstract base class
│   ├── strategy_factory.py    # Strategy generation
│   └── strategy_library.py    # Predefined strategies
├── backtesting/
│   ├── engine.py              # Main backtesting engine
│   ├── metrics.py             # Performance metrics
│   └── comparator.py          # Strategy comparison
├── visualization/
│   ├── charts.py              # Chart generation
│   ├── dashboard.py           # Main dashboard
│   └── report_generator.py    # PDF/HTML reports
├── utils/
│   ├── config.py              # Configuration
│   ├── logger.py              # Logging
│   └── helpers.py             # Utility functions
├── tests/
│   └── ...                    # Unit tests
├── main.py                    # Entry point
└── requirements.txt           # Dependencies
```

---

## Phase 2: Core Data Infrastructure (Week 2-3)

### 2.1 Data Ingestion Module

**File: `data/ingestion.py`**

```python
import yfinance as yf
import pandas as pd
from typing import List, Tuple
from datetime import datetime, timedelta

class MarketDataIngester:
    """Fetches and preprocesses market data"""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
    
    def fetch_historical_data(self) -> dict:
        """Fetch historical OHLCV data"""
        for symbol in self.symbols:
            df = yf.download(symbol, start=self.start_date, end=self.end_date)
            self.data[symbol] = df
        return self.data
    
    def add_technical_indicators(self, symbol: str) -> pd.DataFrame:
        """Add technical indicators (moving averages, RSI, etc.)"""
        df = self.data[symbol].copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        return df.dropna()
    
    def get_processed_data(self) -> dict:
        """Return fully processed data with indicators"""
        processed = {}
        for symbol in self.symbols:
            processed[symbol] = self.add_technical_indicators(symbol)
        return processed
```

### 2.2 Market State Space Model

**File: `models/state_space.py`**

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class MarketTrend(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

class Volatility(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class MarketState:
    """Represents a discrete market state"""
    timestamp: str
    price: float
    trend: MarketTrend
    volatility: Volatility
    rsi: float
    macd_signal: str  # 'buy', 'sell', 'neutral'
    volume_trend: str  # 'increasing', 'decreasing', 'stable'
    
    def __hash__(self):
        return hash((self.trend, self.volatility, self.macd_signal, self.volume_trend))
    
    def __eq__(self, other):
        return (self.trend == other.trend and 
                self.volatility == other.volatility and
                self.macd_signal == other.macd_signal and
                self.volume_trend == other.volume_trend)

class StateSpaceBuilder:
    """Converts continuous market data into discrete states"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def classify_trend(self, row) -> MarketTrend:
        """Classify market trend based on moving averages"""
        if row['SMA_20'] > row['SMA_50'] * 1.02:
            return MarketTrend.STRONG_BULLISH
        elif row['SMA_20'] > row['SMA_50']:
            return MarketTrend.BULLISH
        elif row['SMA_20'] < row['SMA_50'] * 0.98:
            return MarketTrend.STRONG_BEARISH
        elif row['SMA_20'] < row['SMA_50']:
            return MarketTrend.BEARISH
        else:
            return MarketTrend.NEUTRAL
    
    def classify_volatility(self, row) -> Volatility:
        """Classify volatility using Bollinger Bands"""
        bb_width = (row['BB_upper'] - row['BB_lower']) / row['BB_middle']
        if bb_width > 0.1:
            return Volatility.HIGH
        elif bb_width > 0.05:
            return Volatility.MEDIUM
        else:
            return Volatility.LOW
    
    def classify_macd(self, row) -> str:
        """MACD signal classification"""
        if row['MACD'] > row['Signal']:
            return 'buy'
        elif row['MACD'] < row['Signal']:
            return 'sell'
        else:
            return 'neutral'
    
    def classify_volume(self, row) -> str:
        """Volume trend classification"""
        if row['Volume'] > row['Volume_SMA'] * 1.2:
            return 'increasing'
        elif row['Volume'] < row['Volume_SMA'] * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def build_states(self) -> List[MarketState]:
        """Convert dataframe to list of MarketState objects"""
        states = []
        for idx, row in self.data.iterrows():
            state = MarketState(
                timestamp=str(idx),
                price=row['Close'],
                trend=self.classify_trend(row),
                volatility=self.classify_volatility(row),
                rsi=row['RSI'],
                macd_signal=self.classify_macd(row),
                volume_trend=self.classify_volume(row)
            )
            states.append(state)
        return states
```

---

## Phase 3: Graph-Based Market Modeling (Week 3-4)

### 3.1 Market Graph Construction

**File: `models/market_graph.py`**

```python
import networkx as nx
from typing import List, Dict, Tuple
import numpy as np

class MarketGraph:
    """Represents market as a weighted directed graph"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.state_to_node = {}
        self.node_to_state = {}
        self.node_counter = 0
    
    def add_state(self, state: MarketState) -> int:
        """Add a state as a node"""
        if state not in self.state_to_node:
            node_id = self.node_counter
            self.graph.add_node(node_id, state=state, price=state.price)
            self.state_to_node[state] = node_id
            self.node_to_state[node_id] = state
            self.node_counter += 1
            return node_id
        return self.state_to_node[state]
    
    def add_transition(self, from_state: MarketState, to_state: MarketState, 
                      action: str, profit: float, probability: float = 1.0):
        """Add edge representing state transition"""
        from_node = self.add_state(from_state)
        to_node = self.add_state(to_state)
        
        self.graph.add_edge(
            from_node, to_node,
            action=action,
            profit=profit,
            probability=probability,
            weight=-profit  # Negative for shortest path = max profit
        )
    
    def build_from_states(self, states: List[MarketState], actions: List[str] = ['buy', 'sell', 'hold']):
        """Build graph from sequence of states"""
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            
            # Calculate profit for different actions
            price_change = next_state.price - current_state.price
            
            for action in actions:
                profit = self._calculate_profit(action, price_change, current_state.price)
                self.add_transition(current_state, next_state, action, profit)
    
    def _calculate_profit(self, action: str, price_change: float, current_price: float) -> float:
        """Calculate profit based on action"""
        if action == 'buy':
            # Profit realized in next step if price goes up
            return price_change if price_change > 0 else 0
        elif action == 'sell':
            # Profit if we had bought previously (simplified)
            return -price_change if price_change < 0 else 0
        else:  # hold
            return 0
    
    def get_optimal_path_dijkstra(self, start_node: int, end_node: int) -> Tuple[List[int], float]:
        """Find optimal profit path using Dijkstra"""
        try:
            path = nx.dijkstra_path(self.graph, start_node, end_node, weight='weight')
            length = nx.dijkstra_path_length(self.graph, start_node, end_node, weight='weight')
            return path, -length  # Convert back to profit
        except nx.NetworkXNoPath:
            return [], 0
    
    def get_optimal_path_bellman_ford(self, start_node: int) -> Dict[int, float]:
        """Find optimal profits from start node to all others (handles negative cycles)"""
        try:
            distances = nx.single_source_bellman_ford_path_length(
                self.graph, start_node, weight='weight'
            )
            # Convert to profits
            profits = {node: -dist for node, dist in distances.items()}
            return profits
        except nx.NetworkXError as e:
            print(f"Negative cycle detected: {e}")
            return {}
    
    def find_all_paths_bfs(self, start_node: int, max_depth: int = 10) -> List[List[int]]:
        """Find all possible paths using BFS (limited depth)"""
        paths = []
        queue = [(start_node, [start_node])]
        
        while queue:
            node, path = queue.pop(0)
            
            if len(path) >= max_depth:
                paths.append(path)
                continue
            
            for neighbor in self.graph.neighbors(node):
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))
        
        return paths
    
    def detect_profitable_cycles(self) -> List[List[int]]:
        """Detect cycles that yield positive profit (arbitrage opportunities)"""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            profitable_cycles = []
            
            for cycle in cycles:
                total_profit = 0
                for i in range(len(cycle)):
                    from_node = cycle[i]
                    to_node = cycle[(i + 1) % len(cycle)]
                    if self.graph.has_edge(from_node, to_node):
                        profit = self.graph[from_node][to_node]['profit']
                        total_profit += profit
                
                if total_profit > 0:
                    profitable_cycles.append(cycle)
            
            return profitable_cycles
        except:
            return []
```

---

## Phase 4: Dynamic Programming Strategies (Week 4-5)

### 4.1 Optimal Trading with DP

**File: `algorithms/dynamic_programming/optimal_trading.py`**

```python
import numpy as np
from typing import List, Tuple, Dict

class DPTradingOptimizer:
    """Dynamic Programming for optimal buy/sell decisions"""
    
    def __init__(self, prices: List[float], transaction_cost: float = 0.001):
        self.prices = prices
        self.n = len(prices)
        self.transaction_cost = transaction_cost
        self.dp = None
        self.actions = None
    
    def solve_max_profit_k_transactions(self, k: int) -> float:
        """
        DP[i][j][0] = max profit at day i with j transactions, currently NOT holding
        DP[i][j][1] = max profit at day i with j transactions, currently holding
        """
        if self.n == 0:
            return 0
        
        # If k >= n/2, we can make as many transactions as we want
        if k >= self.n // 2:
            return self.solve_unlimited_transactions()
        
        # Initialize DP table
        dp = np.zeros((self.n, k + 1, 2))
        
        for j in range(k + 1):
            dp[0][j][0] = 0
            dp[0][j][1] = -self.prices[0] * (1 + self.transaction_cost)
        
        for i in range(1, self.n):
            for j in range(k + 1):
                # Not holding: max of (didn't hold yesterday, sold today)
                dp[i][j][0] = max(
                    dp[i-1][j][0],
                    dp[i-1][j][1] + self.prices[i] * (1 - self.transaction_cost)
                )
                
                # Holding: max of (held yesterday, bought today)
                if j > 0:
                    dp[i][j][1] = max(
                        dp[i-1][j][1],
                        dp[i-1][j-1][0] - self.prices[i] * (1 + self.transaction_cost)
                    )
        
        self.dp = dp
        return dp[self.n-1][k][0]
    
    def solve_unlimited_transactions(self) -> float:
        """DP for unlimited transactions (greedy equivalent)"""
        profit = 0
        for i in range(1, self.n):
            if self.prices[i] > self.prices[i-1]:
                profit += (self.prices[i] - self.prices[i-1]) * (1 - 2 * self.transaction_cost)
        return profit
    
    def solve_with_cooldown(self) -> float:
        """DP with cooldown period (can't buy immediately after sell)"""
        if self.n == 0:
            return 0
        
        # States: hold, sold (cooldown), rest
        hold = -self.prices[0]
        sold = 0
        rest = 0
        
        for i in range(1, self.n):
            prev_hold = hold
            prev_sold = sold
            prev_rest = rest
            
            hold = max(prev_hold, prev_rest - self.prices[i])
            sold = prev_hold + self.prices[i]
            rest = max(prev_rest, prev_sold)
        
        return max(sold, rest)
    
    def get_optimal_actions(self, k: int) -> List[str]:
        """Backtrack to get the actual buy/sell actions"""
        if self.dp is None:
            self.solve_max_profit_k_transactions(k)
        
        actions = ['hold'] * self.n
        i, j = self.n - 1, k
        holding = 0  # Start from not holding
        
        while i > 0 and j > 0:
            if holding == 0:
                # We're not holding
                if self.dp[i][j][0] == self.dp[i-1][j][1] + self.prices[i] * (1 - self.transaction_cost):
                    actions[i] = 'sell'
                    holding = 1
                i -= 1
            else:
                # We're holding
                if self.dp[i][j][1] == self.dp[i-1][j-1][0] - self.prices[i] * (1 + self.transaction_cost):
                    actions[i] = 'buy'
                    holding = 0
                    j -= 1
                i -= 1
        
        return actions
    
    def analyze_complexity(self) -> Dict:
        """Return time/space complexity analysis"""
        return {
            'time_complexity': f'O(n * k) where n={self.n}, k=transactions',
            'space_complexity': f'O(n * k * 2) = O(n * k)',
            'optimal': True,
            'algorithm': 'Dynamic Programming'
        }
```

### 4.2 Portfolio Allocation DP (Knapsack)

**File: `algorithms/dynamic_programming/portfolio_dp.py`**

```python
from typing import List, Tuple
import numpy as np

class PortfolioKnapsackDP:
    """Solve portfolio allocation as a knapsack problem"""
    
    def __init__(self, assets: List[dict], capital: float):
        """
        assets: List of {'symbol': str, 'cost': float, 'expected_return': float, 'risk': float}
        capital: Available capital
        """
        self.assets = assets
        self.capital = int(capital * 100)  # Convert to cents for integer DP
        self.n = len(assets)
        self.dp = None
    
    def solve_01_knapsack(self) -> Tuple[float, List[int]]:
        """
        0/1 Knapsack: Each asset can be bought once
        Returns: (max_return, selected_indices)
        """
        # DP table: dp[i][w] = max return using first i assets with capital w
        dp = np.zeros((self.n + 1, self.capital + 1))
        
        for i in range(1, self.n + 1):
            asset = self.assets[i-1]
            cost = int(asset['cost'] * 100)
            return_val = asset['expected_return']
            
            for w in range(self.capital + 1):
                # Don't take this asset
                dp[i][w] = dp[i-1][w]
                
                # Take this asset if we can afford it
                if w >= cost:
                    dp[i][w] = max(dp[i][w], dp[i-1][w-cost] + return_val)
        
        self.dp = dp
        
        # Backtrack to find selected assets
        selected = []
        w = self.capital
        for i in range(self.n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected.append(i-1)
                cost = int(self.assets[i-1]['cost'] * 100)
                w -= cost
        
        return dp[self.n][self.capital], selected[::-1]
    
    def solve_unbounded_knapsack(self) -> Tuple[float, Dict[int, int]]:
        """
        Unbounded Knapsack: Can buy multiple units of each asset
        Returns: (max_return, {asset_index: quantity})
        """
        dp = np.zeros(self.capital + 1)
        
        for w in range(self.capital + 1):
            for i, asset in enumerate(self.assets):
                cost = int(asset['cost'] * 100)
                return_val = asset['expected_return']
                
                if w >= cost:
                    dp[w] = max(dp[w], dp[w-cost] + return_val)
        
        # Backtrack to find quantities
        quantities = {i: 0 for i in range(self.n)}
        w = self.capital
        
        while w > 0:
            for i, asset in enumerate(self.assets):
                cost = int(asset['cost'] * 100)
                return_val = asset['expected_return']
                
                if w >= cost and dp[w] == dp[w-cost] + return_val:
                    quantities[i] += 1
                    w -= cost
                    break
        
        return dp[self.capital], quantities
    
    def solve_with_risk_constraint(self, max_risk: float) -> Tuple[float, List[int]]:
        """
        2D DP: Maximize return subject to capital AND risk constraints
        dp[i][w][r] = max return with first i assets, capital w, risk <= r
        """
        max_risk_int = int(max_risk * 100)
        
        # 3D DP table
        dp = np.full((self.n + 1, self.capital + 1, max_risk_int + 1), -np.inf)
        dp[0][0][0] = 0
        
        for i in range(1, self.n + 1):
            asset = self.assets[i-1]
            cost = int(asset['cost'] * 100)
            return_val = asset['expected_return']
            risk = int(asset['risk'] * 100)
            
            for w in range(self.capital + 1):
                for r in range(max_risk_int + 1):
                    # Don't take this asset
                    dp[i][w][r] = dp[i-1][w][r]
                    
                    # Take this asset
                    if w >= cost and r >= risk:
                        dp[i][w][r] = max(
                            dp[i][w][r],
                            dp[i-1][w-cost][r-risk] + return_val
                        )
        
        max_return = np.max(dp[self.n][self.capital])
        return max_return, []  # Simplified - backtracking similar to above
```

---

## Phase 5: Greedy Algorithms (Week 5)

**File: `algorithms/greedy/threshold_strategy.py`**

```python
from typing import List, Tuple
import numpy as np

class GreedyTradingStrategy:
    """Greedy algorithms for trading"""
    
    def __init__(self, prices: List[float], indicators: dict = None):
        self.prices = prices
        self.indicators = indicators or {}
        self.n = len(prices)
    
    def simple_greedy(self) -> Tuple[float, List[str]]:
        """
        Greedy: Buy when price drops, sell when price rises
        Time: O(n), Space: O(n)
        """
        profit = 0
        actions = ['hold'] * self.n
        holding = False
        buy_price = 0
        
        for i in range(1, self.n):
            if not holding and self.prices[i] > self.prices[i-1]:
                # Buy at previous day's price
                buy_price = self.prices[i-1]
                holding = True
                actions[i-1] = 'buy'
            elif holding and self.prices[i] < self.prices[i-1]:
                # Sell at previous day's price
                profit += (self.prices[i-1] - buy_price)
                holding = False
                actions[i-1] = 'sell'
        
        # Sell at end if still holding
        if holding:
            profit += (self.prices[-1] - buy_price)
            actions[-1] = 'sell'
        
        return profit, actions
    
    def threshold_greedy(self, buy_threshold: float = -0.02, 
                        sell_threshold: float = 0.03) -> Tuple[float, List[str]]:
        """
        Greedy with thresholds: Buy when drop > buy_threshold, 
        sell when gain > sell_threshold
        """
        profit = 0
        actions = ['hold'] * self.n
        holding = False
        buy_price = 0
        
        for i in range(1, self.n):
            price_change = (self.prices[i] - self.prices[i-1]) / self.prices[i-1]
            
            if not holding and price_change <= buy_threshold:
                buy_price = self.prices[i]
                holding = True
                actions[i] = 'buy'
            elif holding:
                gain = (self.prices[i] - buy_price) / buy_price
                if gain >= sell_threshold:
                    profit += (self.prices[i] - buy_price)
                    holding = False
                    actions[i] = 'sell'
        
        if holding:
            profit += (self.prices[-1] - buy_price)
            actions[-1] = 'sell'
        
        return profit, actions
    
    def rsi_greedy(self, rsi_values: List[float], 
                   buy_rsi: float = 30, sell_rsi: float = 70) -> Tuple[float, List[str]]:
        """
        Greedy based on RSI indicator
        Buy when RSI < buy_rsi (oversold)
        Sell when RSI > sell_rsi (overbought)
        """
        profit = 0
        actions = ['hold'] * self.n
        holding = False
        buy_price = 0
        
        for i in range(len(rsi_values)):
            if not holding and rsi_values[i] < buy_rsi:
                buy_price = self.prices[i]
                holding = True
                actions[i] = 'buy'
            elif holding and rsi_values[i] > sell_rsi:
                profit += (self.prices[i] - buy_price)
                holding = False
                actions[i] = 'sell'
        
        if holding:
            profit += (self.prices[-1] - buy_price)
            actions[-1] = 'sell'
        
        return profit, actions
    
    def compare_with_optimal(self, optimal_profit: float, greedy_profit: float) -> dict:
        """Analyze when greedy fails compared to DP optimal"""
        approximation_ratio = greedy_profit / optimal_profit if optimal_profit > 0 else 0
        
        return {
            'greedy_profit': greedy_profit,
            'optimal_profit': optimal_profit,
            'approximation_ratio': approximation_ratio,
            'performance': 'good' if approximation_ratio > 0.8 else 'poor',
            'analysis': self._analyze_failure_cases()
        }
    
    def _analyze_failure_cases(self) -> str:
        """Theoretical analysis of when greedy fails"""
        return """
        Greedy algorithm fails in:
        1. Oscillating markets: Buys/sells too frequently, missing larger trends
        2. Gradual trends: Misses optimal entry/exit points
        3. High transaction costs: Multiple trades reduce profit
        4. Look-ahead bias: Can't see future price movements
        
        Greedy achieves O(n) time but suboptimal profit.
        DP achieves optimal profit with O(n*k) time.
        """
```

---

## Phase 6: Branch & Bound Optimization (Week 6)

**File: `algorithms/optimization/branch_bound.py`**

```python
from typing import List, Tuple, Optional
import heapq
from dataclasses import dataclass, field

@dataclass(order=True)
class Node:
    """Node in the branch & bound search tree"""
    priority: float = field(compare=True)
    level: int = field(compare=False)
    profit: float = field(compare=False)
    cost: float = field(compare=False)
    selected: List[int] = field(compare=False, default_factory=list)
    bound: float = field(compare=False, default=0)

class BranchAndBoundOptimizer:
    """Branch & Bound for strategy optimization"""
    
    def __init__(self, strategies: List[dict], capital: float):
        """
        strategies: [{'id': int, 'cost': float, 'expected_profit': float, 'risk': float}]
        """
        self.strategies = sorted(strategies, 
                                key=lambda x: x['expected_profit']/x['cost'], 
                                reverse=True)
        self.capital = capital
        self.n = len(strategies)
        self.best_solution = None
        self.best_profit = 0
        self.nodes_explored = 0
    
    def calculate_upper_bound(self, node: Node) -> float:
        """
        Calculate upper bound using fractional knapsack relaxation
        This gives the maximum possible profit from this node
        """
        if node.cost >= self.capital:
            return 0
        
        bound = node.profit
        total_cost = node.cost
        level = node.level
        
        # Add items while we can fit them completely
        while level < self.n and total_cost + self.strategies[level]['cost'] <= self.capital:
            total_cost += self.strategies[level]['cost']
            bound += self.strategies[level]['expected_profit']
            level += 1
        
        # Add fractional part of next item
        if level < self.n:
            remaining_capacity = self.capital - total_cost
            bound += (remaining_capacity / self.strategies[level]['cost']) * \
                     self.strategies[level]['expected_profit']
        
        return bound
    
    def solve(self) -> Tuple[float, List[int]]:
        """
        Branch & Bound algorithm to find optimal strategy combination
        Returns: (max_profit, selected_strategy_ids)
        """
        # Priority queue: (negative bound for max-heap, node)
        pq = []
        
        # Root node
        root = Node(
            priority=0,
            level=0,
            profit=0,
            cost=0,
            selected=[],
            bound=0
        )
        root.bound = self.calculate_upper_bound(root)
        root.priority = -root.bound  # Negative for max-heap
        
        heapq.heappush(pq, root)
        
        while pq:
            current = heapq.heappop(pq)
            self.nodes_explored += 1
            
            # Prune if bound is not better than current best
            if current.bound <= self.best_profit:
                continue
            
            # If we've processed all strategies
            if current.level >= self.n:
                continue
            
            strategy = self.strategies[current.level]
            
            # Branch 1: Include this strategy
            if current.cost + strategy['cost'] <= self.capital:
                include_node = Node(
                    priority=0,
                    level=current.level + 1,
                    profit=current.profit + strategy['expected_profit'],
                    cost=current.cost + strategy['cost'],
                    selected=current.selected + [strategy['id']]
                )
                
                # Update best if this is a leaf and better
                if include_node.profit > self.best_profit:
                    self.best_profit = include_node.profit
                    self.best_solution = include_node.selected[:]
                
                # Calculate bound and add to queue if promising
                include_node.bound = self.calculate_upper_bound(include_node)
                if include_node.bound > self.best_profit:
                    include_node.priority = -include_node.bound
                    heapq.heappush(pq, include_node)
            
            # Branch 2: Exclude this strategy
            exclude_node = Node(
                priority=0,
                level=current.level + 1,
                profit=current.profit,
                cost=current.cost,
                selected=current.selected[:]
            )
            
            exclude_node.bound = self.calculate_upper_bound(exclude_node)
            if exclude_node.bound > self.best_profit:
                exclude_node.priority = -exclude_node.bound
                heapq.heappush(pq, exclude_node)
        
        return self.best_profit, self.best_solution
    
    def get_statistics(self) -> dict:
        """Return algorithm performance statistics"""
        return {
            'nodes_explored': self.nodes_explored,
            'nodes_pruned': 2**self.n - self.nodes_explored,
            'pruning_efficiency': 1 - (self.nodes_explored / 2**self.n),
            'time_complexity': f'O(2^n) worst case, but pruned to {self.nodes_explored} nodes',
            'space_complexity': 'O(n) for priority queue'
        }
```

---

## Phase 7: Game Theory Module (Week 7-8)

### 7.1 Minimax Algorithm

**File: `algorithms/game_theory/minimax.py`**

```python
from typing import List, Tuple, Optional
import numpy as np

class MinimaxTrader:
    """Minimax algorithm for adversarial trading"""
    
    def __init__(self, depth: int = 5):
        self.depth = depth
        self.nodes_evaluated = 0
    
    def minimax(self, state: dict, depth: int, is_maximizing: bool, 
                alpha: float = -np.inf, beta: float = np.inf) -> Tuple[float, str]:
        """
        Minimax with alpha-beta pruning
        
        state: {
            'price': float,
            'position': str ('long', 'short', 'neutral'),
            'cash': float,
            'shares': int
        }
        
        Returns: (score, action)
        """
        self.nodes_evaluated += 1
        
        # Terminal state or max depth
        if depth == 0 or self.is_terminal(state):
            return self.evaluate(state), 'hold'
        
        actions = self.get_possible_actions(state)
        
        if is_maximizing:
            # Our turn (maximize profit)
            max_eval = -np.inf
            best_action = 'hold'
            
            for action in actions:
                next_state = self.apply_action(state, action)
                eval_score, _ = self.minimax(next_state, depth - 1, False, alpha, beta)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            
            return max_eval, best_action
        
        else:
            # Opponent's turn (minimize our profit)
            min_eval = np.inf
            best_action = 'hold'
            
            for action in actions:
                next_state = self.apply_action(state, action, opponent=True)
                eval_score, _ = self.minimax(next_state, depth - 1, True, alpha, beta)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            return min_eval, best_action
    
    def evaluate(self, state: dict) -> float:
        """
        Heuristic evaluation function
        Returns a score representing how good this state is for us
        """
        score = state['cash']
        
        # Add value of current position
        if state['position'] == 'long':
            score += state['shares'] * state['price']
        elif state['position'] == 'short':
            score += state['shares'] * state['price']  # Profit from short
        
        # Penalize risk
        volatility_penalty = state.get('volatility', 0) * 0.1
        score -= volatility_penalty
        
        return score
    
    def is_terminal(self, state: dict) -> bool:
        """Check if this is a terminal state"""
        return state['cash'] <= 0 or state['price'] <= 0
    
    def get_possible_actions(self, state: dict) -> List[str]:
        """Get all possible actions from this state"""
        actions = ['hold']
        
        if state['position'] == 'neutral' and state['cash'] > state['price']:
            actions.extend(['buy', 'short'])
        elif state['position'] == 'long':
            actions.append('sell')
        elif state['position'] == 'short':
            actions.append('cover')
        
        return actions
    
    def apply_action(self, state: dict, action: str, opponent: bool = False) -> dict:
        """
        Apply an action and return new state
        If opponent=True, simulate opponent's market-moving action
        """
        new_state = state.copy()
        
        if opponent:
            # Opponent actions affect price
            price_impact = np.random.uniform(-0.02, 0.02)
            new_state['price'] *= (1 + price_impact)
        else:
            # Our actions
            if action == 'buy':
                shares = int(new_state['cash'] / new_state['price'])
                new_state['shares'] = shares
                new_state['cash'] -= shares * new_state['price']
                new_state['position'] = 'long'
            
            elif action == 'sell':
                new_state['cash'] += new_state['shares'] * new_state['price']
                new_state['shares'] = 0
                new_state['position'] = 'neutral'
            
            elif action == 'short':
                shares = int(new_state['cash'] / new_state['price'])
                new_state['shares'] = shares
                new_state['position'] = 'short'
            
            elif action == 'cover':
                new_state['cash'] += new_state['shares'] * new_state['price']
                new_state['shares'] = 0
                new_state['position'] = 'neutral'
        
        return new_state
    
    def get_best_action(self, current_state: dict) -> str:
        """Get the best action for current state"""
        _, action = self.minimax(current_state, self.depth, True)
        return action
```

### 7.2 Nash Equilibrium

**File: `algorithms/game_theory/nash_equilibrium.py`**

```python
import numpy as np
from scipy.optimize import linprog
from typing import List, Tuple

class NashEquilibriumFinder:
    """Find Nash Equilibrium in trading games"""
    
    def __init__(self, payoff_matrix_player1: np.ndarray, 
                 payoff_matrix_player2: np.ndarray):
        """
        payoff_matrix_player1: Player 1's payoffs (rows = P1 strategies, cols = P2 strategies)
        payoff_matrix_player2: Player 2's payoffs
        """
        self.payoff1 = payoff_matrix_player1
        self.payoff2 = payoff_matrix_player2
        self.n_strategies_p1 = payoff_matrix_player1.shape[0]
        self.n_strategies_p2 = payoff_matrix_player1.shape[1]
    
    def find_pure_nash(self) -> List[Tuple[int, int]]:
        """
        Find pure strategy Nash Equilibria
        Returns list of (strategy_p1, strategy_p2) tuples
        """
        equilibria = []
        
        for i in range(self.n_strategies_p1):
            for j in range(self.n_strategies_p2):
                # Check if (i, j) is a Nash equilibrium
                is_nash = True
                
                # Check if P1 wants to deviate
                for i_prime in range(self.n_strategies_p1):
                    if self.payoff1[i_prime][j] > self.payoff1[i][j]:
                        is_nash = False
                        break
                
                # Check if P2 wants to deviate
                if is_nash:
                    for j_prime in range(self.n_strategies_p2):
                        if self.payoff2[i][j_prime] > self.payoff2[i][j]:
                            is_nash = False
                            break
                
                if is_nash:
                    equilibria.append((i, j))
        
        return equilibria
    
    def find_mixed_nash_2x2(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find mixed strategy Nash Equilibrium for 2x2 games
        Returns: (prob_vector_p1, prob_vector_p2)
        """
        if self.n_strategies_p1 != 2 or self.n_strategies_p2 != 2:
            raise ValueError("This method only works for 2x2 games")
        
        # For Player 1: find probabilities that make P2 indifferent
        # P2's expected payoff from strategy 0: p * payoff2[0][0] + (1-p) * payoff2[1][0]
        # P2's expected payoff from strategy 1: p * payoff2[0][1] + (1-p) * payoff2[1][1]
        # Set equal and solve for p
        
        numerator = self.payoff2[1][1] - self.payoff2[1][0]
        denominator = (self.payoff2[0][0] - self.payoff2[1][0] - 
                      self.payoff2[0][1] + self.payoff2[1][1])
        
        if denominator != 0:
            p1_prob = numerator / denominator
            p1_prob = np.clip(p1_prob, 0, 1)
        else:
            p1_prob = 0.5
        
        # For Player 2: find probabilities that make P1 indifferent
        numerator = self.payoff1[1][1] - self.payoff1[0][1]
        denominator = (self.payoff1[0][0] - self.payoff1[0][1] - 
                      self.payoff1[1][0] + self.payoff1[1][1])
        
        if denominator != 0:
            p2_prob = numerator / denominator
            p2_prob = np.clip(p2_prob, 0, 1)
        else:
            p2_prob = 0.5
        
        prob_p1 = np.array([p1_prob, 1 - p1_prob])
        prob_p2 = np.array([p2_prob, 1 - p2_prob])
        
        return prob_p1, prob_p2
    
    def example_trading_game(self) -> dict:
        """
        Example: Two traders competing
        Strategies: Aggressive Buy, Conservative Buy, Hold, Aggressive Sell
        """
        # Define payoff matrices
        payoff1 = np.array([
            [3, 5, 4, 2],    # Aggressive Buy
            [4, 3, 3, 1],    # Conservative Buy
            [2, 2, 0, -1],   # Hold
            [-1, 0, 1, 2]    # Aggressive Sell
        ])
        
        # Symmetric game (zero-sum approximation)
        payoff2 = -payoff1
        
        self.payoff1 = payoff1
        self.payoff2 = payoff2
        
        pure_nash = self.find_pure_nash()
        
        return {
            'payoff_matrix_p1': payoff1,
            'payoff_matrix_p2': payoff2,
            'pure_nash_equilibria': pure_nash,
            'interpretation': 'Strategies are indexed as: 0=Aggressive Buy, 1=Conservative Buy, 2=Hold, 3=Aggressive Sell'
        }
```

---

## Phase 8: Backtesting Engine (Week 8-9)

**File: `backtesting/engine.py`**

```python
from typing import List, Dict, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class Trade:
    timestamp: str
    action: str  # 'buy', 'sell', 'hold'
    price: float
    quantity: int
    value: float

@dataclass
class BacktestResult:
    strategy_name: str
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    trades: List[Trade]
    equity_curve: pd.Series
    metrics: Dict

class BacktestEngine:
    """Comprehensive backtesting engine"""
    
    def __init__(self, initial_capital: float = 10000, 
                 transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
    
    def run_backtest(self, data: pd.DataFrame, 
                    strategy_func: Callable,
                    strategy_name: str) -> BacktestResult:
        """
        Run backtest for a given strategy
        
        data: DataFrame with OHLCV data and indicators
        strategy_func: Function that takes (data, current_index) and returns action
        """
        capital = self.initial_capital
        position = 0  # Number of shares held
        trades = []
        equity_curve = []
        
        for i in range(len(data)):
            current_price = data.iloc[i]['Close']
            
            # Get strategy action
            action = strategy_func(data, i)
            
            # Execute action
            if action == 'buy' and capital >= current_price:
                # Buy as many shares as possible
                quantity = int(capital / (current_price * (1 + self.transaction_cost)))
                if quantity > 0:
                    cost = quantity * current_price * (1 + self.transaction_cost)
                    capital -= cost
                    position += quantity
                    
                    trades.append(Trade(
                        timestamp=str(data.index[i]),
                        action='buy',
                        price=current_price,
                        quantity=quantity,
                        value=cost
                    ))
            
            elif action == 'sell' and position > 0:
                # Sell all shares
                revenue = position * current_price * (1 - self.transaction_cost)
                capital += revenue
                
                trades.append(Trade(
                    timestamp=str(data.index[i]),
                    action='sell',
                    price=current_price,
                    quantity=position,
                    value=revenue
                ))
                
                position = 0
            
            # Calculate current equity
            current_equity = capital + (position * current_price)
            equity_curve.append(current_equity)
        
        # Close any remaining position
        if position > 0:
            final_price = data.iloc[-1]['Close']
            capital += position * final_price * (1 - self.transaction_cost)
            trades.append(Trade(
                timestamp=str(data.index[-1]),
                action='sell',
                price=final_price,
                quantity=position,
                value=position * final_price
            ))
        
        # Calculate metrics
        equity_series = pd.Series(equity_curve, index=data.index)
        metrics = self._calculate_metrics(equity_series, trades)
        
        return BacktestResult(
            strategy_name=strategy_name,
            total_return=(capital - self.initial_capital) / self.initial_capital,
            total_trades=len(trades),
            winning_trades=metrics['winning_trades'],
            losing_trades=metrics['losing_trades'],
            max_drawdown=metrics['max_drawdown'],
            sharpe_ratio=metrics['sharpe_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            trades=trades,
            equity_curve=equity_series,
            metrics=metrics
        )
    
    def _calculate_metrics(self, equity_curve: pd.Series, 
                          trades: List[Trade]) -> Dict:
        """Calculate performance metrics"""
        
        # Returns
        returns = equity_curve.pct_change().dropna()
        
        # Winning/Losing trades
        winning_trades = 0
        losing_trades = 0
        trade_returns = []
        
        for i in range(1, len(trades), 2):  # Pairs of buy-sell
            if i < len(trades):
                buy_trade = trades[i-1]
                sell_trade = trades[i]
                if sell_trade.action == 'sell':
                    profit = sell_trade.value - buy_trade.value
                    trade_returns.append(profit / buy_trade.value)
                    if profit > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
        
        # Maximum Drawdown
        cumulative = equity_curve / equity_curve.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (annualized)
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Sortino Ratio (uses downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        # Win Rate
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        return {
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'total_return': (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0],
            'volatility': returns.std() * np.sqrt(252),
            'avg_trade_return': np.mean(trade_returns) if trade_returns else 0
        }
    
    def compare_strategies(self, results: List[BacktestResult]) -> pd.DataFrame:
        """Compare multiple strategy results"""
        comparison = []
        
        for result in results:
            comparison.append({
                'Strategy': result.strategy_name,
                'Total Return': f"{result.total_return * 100:.2f}%",
                'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                'Max Drawdown': f"{result.max_drawdown * 100:.2f}%",
                'Win Rate': f"{result.metrics['win_rate'] * 100:.2f}%",
                'Total Trades': result.total_trades,
                'Avg Trade Return': f"{result.metrics['avg_trade_return'] * 100:.2f}%"
            })
        
        return pd.DataFrame(comparison)
```

---

## Phase 9: Visualization & Dashboard (Week 9-10)

**File: `visualization/dashboard.py`**

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

class TradingDashboard:
    """Interactive dashboard for visualizing results"""
    
    def __init__(self):
        self.figures = {}
    
    def plot_equity_curves(self, results: List[BacktestResult]):
        """Plot equity curves for multiple strategies"""
        fig = go.Figure()
        
        for result in results:
            fig.add_trace(go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                name=result.strategy_name,
                mode='lines'
            ))
        
        fig.update_layout(
            title='Equity Curves Comparison',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_drawdown(self, equity_curve: pd.Series, strategy_name: str):
        """Plot drawdown chart"""
        cumulative = equity_curve / equity_curve.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f'Drawdown - {strategy_name}',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_white'
        )
        
        return fig
    
    def plot_algorithm_comparison_bar(self, comparison_df: pd.DataFrame):
        """Bar chart comparing algorithm performance"""
        fig = go.Figure()
        
        metrics = ['Total Return', 'Sharpe Ratio', 'Win Rate']
        
        for metric in metrics:
            # Extract numeric values
            values = comparison_df[metric].str.replace('%', '').astype(float)
            
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Strategy'],
                y=values,
                text=comparison_df[metric],
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Strategy Comparison',
            xaxis_title='Strategy',
            yaxis_title='Value',
            barmode='group',
            template='plotly_white'
        )
        
        return fig
    
    def plot_state_space_graph(self, market_graph):
        """Visualize market state-space graph"""
        import networkx as nx
        
        G = market_graph.graph
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Extract edge info
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Extract node info
        node_x = []
        node_y = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            state = market_graph.node_to_state.get(node)
            if state:
                node_text.append(f"Trend: {state.trend.value}<br>Vol: {state.volatility.value}")
            else:
                node_text.append(f"Node {node}")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=10,
                color='lightblue',
                line_width=2
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title='Market State-Space Graph',
            showlegend=False,
            hovermode='closest',
            template='plotly_white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def create_streamlit_app(self, results: List[BacktestResult], 
                            comparison_df: pd.DataFrame):
        """Create Streamlit dashboard"""
        st.title("Algorithmic Trading Strategy Engine")
        st.subheader("DAA-Based Trading System Analysis")
        
        # Sidebar for strategy selection
        st.sidebar.header("Configuration")
        selected_strategies = st.sidebar.multiselect(
            "Select Strategies to Compare",
            [r.strategy_name for r in results],
            default=[r.strategy_name for r in results[:3]]
        )
        
        # Filter results
        filtered_results = [r for r in results if r.strategy_name in selected_strategies]
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs([
            "Performance Overview", 
            "Detailed Analysis", 
            "Algorithm Comparison",
            "Theoretical Analysis"
        ])
        
        with tab1:
            st.header("Performance Overview")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Strategy", 
                         max(filtered_results, key=lambda x: x.total_return).strategy_name)
            with col2:
                st.metric("Max Return", 
                         f"{max(r.total_return for r in filtered_results) * 100:.2f}%")
            with col3:
                st.metric("Best Sharpe", 
                         f"{max(r.sharpe_ratio for r in filtered_results):.2f}")
            with col4:
                st.metric("Total Strategies", len(filtered_results))
            
            # Equity curves
            st.plotly_chart(self.plot_equity_curves(filtered_results), 
                          use_container_width=True)
        
        with tab2:
            st.header("Detailed Analysis")
            
            selected_strategy = st.selectbox(
                "Select Strategy for Detailed View",
                [r.strategy_name for r in filtered_results]
            )
            
            strategy_result = next(r for r in filtered_results 
                                  if r.strategy_name == selected_strategy)
            
            # Drawdown
            st.plotly_chart(
                self.plot_drawdown(strategy_result.equity_curve, selected_strategy),
                use_container_width=True
            )
            
            # Trade log
            st.subheader("Trade Log")
            trade_df = pd.DataFrame([
                {
                    'Timestamp': t.timestamp,
                    'Action': t.action,
                    'Price': f"${t.price:.2f}",
                    'Quantity': t.quantity,
                    'Value': f"${t.value:.2f}"
                }
                for t in strategy_result.trades[:50]  # Show first 50
            ])
            st.dataframe(trade_df)
        
        with tab3:
            st.header("Algorithm Comparison")
            st.dataframe(comparison_df)
            st.plotly_chart(
                self.plot_algorithm_comparison_bar(comparison_df),
                use_container_width=True
            )
        
        with tab4:
            st.header("Theoretical Analysis")
            
            st.subheader("Complexity Analysis")
            complexity_table = pd.DataFrame([
                {'Algorithm': 'Dynamic Programming', 'Time': 'O(n*k)', 'Space': 'O(n*k)', 'Optimality': 'Optimal'},
                {'Algorithm': 'Greedy', 'Time': 'O(n)', 'Space': 'O(1)', 'Optimality': 'Approximation'},
                {'Algorithm': 'Branch & Bound', 'Time': 'O(2^n) pruned', 'Space': 'O(n)', 'Optimality': 'Optimal'},
                {'Algorithm': 'Minimax', 'Time': 'O(b^d)', 'Space': 'O(bd)', 'Optimality': 'Game-optimal'},
            ])
            st.table(complexity_table)
            
            st.subheader("When Each Algorithm Excels")
            st.write("""
            - **Dynamic Programming**: Best for optimal sequential decisions with overlapping subproblems
            - **Greedy**: Fast execution when local optimality suffices
            - **Branch & Bound**: Optimal portfolio allocation with constraint satisfaction
            - **Minimax**: Adversarial scenarios with competing agents
            - **Graph Algorithms**: State-space exploration and path optimization
            """)
```

---

## Phase 10: Integration & Testing (Week 10-11)

**File: `main.py`**

```python
import sys
from data.ingestion import MarketDataIngester
from models.state_space import StateSpaceBuilder
from models.market_graph import MarketGraph
from algorithms.dynamic_programming.optimal_trading import DPTradingOptimizer
from algorithms.greedy.threshold_strategy import GreedyTradingStrategy
from backtesting.engine import BacktestEngine
from visualization.dashboard import TradingDashboard
import pandas as pd

def main():
    print("=== Algorithmic Trading Strategy Engine ===\n")
    
    # 1. Data Ingestion
    print("Step 1: Fetching market data...")
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    ingester = MarketDataIngester(
        symbols=symbols,
        start_date='2020-01-01',
        end_date='2024-01-01'
    )
    
    ingester.fetch_historical_data()
    processed_data = ingester.get_processed_data()
    print(f"✓ Fetched data for {len(symbols)} symbols\n")
    
    # 2. Build State Space
    print("Step 2: Building state space...")
    symbol = 'AAPL'
    data = processed_data[symbol]
    state_builder = StateSpaceBuilder(data)
    states = state_builder.build_states()
    print(f"✓ Created {len(states)} discrete states\n")
    
    # 3. Build Market Graph
    print("Step 3: Constructing market graph...")
    market_graph = MarketGraph()
    market_graph.build_from_states(states)
    print(f"✓ Graph has {market_graph.graph.number_of_nodes()} nodes and {market_graph.graph.number_of_edges()} edges\n")
    
    # 4. Run Algorithms
    print("Step 4: Running trading algorithms...")
    prices = data['Close'].values.tolist()
    
    # DP Algorithm
    dp_optimizer = DPTradingOptimizer(prices)
    dp_profit = dp_optimizer.solve_max_profit_k_transactions(k=5)
    print(f"✓ DP Algorithm: Max profit = ${dp_profit:.2f}")
    
    # Greedy Algorithm
    greedy_strategy = GreedyTradingStrategy(prices)
    greedy_profit, _ = greedy_strategy.simple_greedy()
    print(f"✓ Greedy Algorithm: Profit = ${greedy_profit:.2f}")
    
    # Comparison
    comparison = greedy_strategy.compare_with_optimal(dp_profit, greedy_profit)
    print(f"✓ Approximation ratio: {comparison['approximation_ratio']:.2%}\n")
    
    # 5. Backtesting
    print("Step 5: Running backtests...")
    backtest_engine = BacktestEngine(initial_capital=10000)
    
    # Define strategy functions
    def dp_strategy(data, i):
        # Simplified DP strategy
        if i > 0 and data.iloc[i]['Close'] > data.iloc[i-1]['Close']:
            return 'buy'
        elif i > 0 and data.iloc[i]['Close'] < data.iloc[i-1]['Close'] * 0.95:
            return 'sell'
        return 'hold'
    
    def greedy_strategy_func(data, i):
        if i > 0:
            rsi = data.iloc[i]['RSI']
            if rsi < 30:
                return 'buy'
            elif rsi > 70:
                return 'sell'
        return 'hold'
    
    # Run backtests
    dp_result = backtest_engine.run_backtest(data, dp_strategy, "DP Strategy")
    greedy_result = backtest_engine.run_backtest(data, greedy_strategy_func, "Greedy RSI Strategy")
    
    print(f"✓ DP Strategy: {dp_result.total_return * 100:.2f}% return")
    print(f"✓ Greedy Strategy: {greedy_result.total_return * 100:.2f}% return\n")
    
    # 6. Generate Comparison Report
    print("Step 6: Generating comparison report...")
    results = [dp_result, greedy_result]
    comparison_df = backtest_engine.compare_strategies(results)
    print("\n" + comparison_df.to_string(index=False) + "\n")
    
    # 7. Visualization
    print("Step 7: Creating visualizations...")
    dashboard = TradingDashboard()
    
    # Save plots
    equity_fig = dashboard.plot_equity_curves(results)
    equity_fig.write_html("output/equity_curves.html")
    print("✓ Saved equity_curves.html")
    
    drawdown_fig = dashboard.plot_drawdown(dp_result.equity_curve, "DP Strategy")
    drawdown_fig.write_html("output/drawdown.html")
    print("✓ Saved drawdown.html")
    
    print("\n=== Execution Complete ===")
    print("Launch Streamlit dashboard: streamlit run dashboard_app.py")

if __name__ == "__main__":
    main()
```

---

## Deliverables Checklist

- [ ] Working code for all 7 algorithm types
- [ ] Backtesting framework with multiple strategies
- [ ] Interactive dashboard (Streamlit)
- [ ] Comparison tables and visualizations
- [ ] Documentation of complexity analysis
- [ ] Example datasets and results
- [ ] README with setup instructions
- [ ] (Optional) Docker container for easy deployment
- [ ] (Optional) Jupyter notebooks with explanations
- [ ] Final presentation/report

## Timeline Summary

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1-2 | Setup | Project structure, data pipeline |
| 2-3 | Data | State space model, market graph |
| 3-4 | Graphs | Graph algorithms implementation |
| 4-5 | DP | Dynamic programming strategies |
| 5 | Greedy | Greedy algorithms + comparison |
| 6 | Optimization | Branch & Bound, knapsack |
| 7-8 | Game Theory | Minimax, Nash equilibrium |
| 8-9 | Backtesting | Complete testing framework |
| 9-10 | Visualization | Dashboard and reports |
| 10-11 | Integration | Testing, documentation, polish |

## Next Steps

1. Set up Python environment with requirements
2. Implement data ingestion module
3. Build state space representation
4. Start with DP algorithms (most important)
5. Add other algorithms incrementally
6. Test each module independently
7. Integrate and create visualizations
8. Prepare final presentation

This gives you a complete, production-ready implementation roadmap!
