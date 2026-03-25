"""
NASDAQ Stock Streak Screener & Analysis Dashboard
===================================================
Screens NASDAQ stocks for consecutive price streaks and analyzes
subsequent returns vs NASDAQ Composite Index.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import re
import warnings
import os
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# ─── Password Protection ─────────────────────────────────────────────────────
# Use a unique key so no other widget (e.g. dropdowns) can overwrite auth state
_AUTH_PASSWORD_KEY = "_dashboard_auth_password"

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        pwd = st.session_state.get(_AUTH_PASSWORD_KEY, "")
        # Only validate non-empty input so dropdown/other reruns don't trigger false "incorrect"
        if not pwd:
            return
        if pwd == os.getenv("DASHBOARD_PASSWORD", "dcr"):
            st.session_state["password_correct"] = True
            if _AUTH_PASSWORD_KEY in st.session_state:
                del st.session_state[_AUTH_PASSWORD_KEY]
        else:
            st.session_state["password_correct"] = False
    
    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Enter password", type="password", on_change=password_entered, key=_AUTH_PASSWORD_KEY
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.text_input(
            "Enter password", type="password", on_change=password_entered, key=_AUTH_PASSWORD_KEY
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NASDAQ Streak Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Check password before showing dashboard
if not check_password():
    st.stop()

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .stApp { font-family: 'DM Sans', sans-serif; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #e0e0e0;
    }
    .metric-card .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        margin-top: 4px;
    }
    
    .section-header {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        color: #e0e0e0;
        border-left: 4px solid #00d4ff;
        padding-left: 12px;
        margin: 24px 0 12px 0;
    }
    
    div[data-testid="stTabs"] button {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
    }
    
    .up-streak { color: #00c853; font-weight: 600; }
    .down-streak { color: #ff1744; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ────────────────────────────────────────────────────────────
# Prefer script directory, then current working directory (cloud/runtime-safe)
SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = Path.cwd()
SEARCH_DIRS = [SCRIPT_DIR] if SCRIPT_DIR == WORK_DIR else [SCRIPT_DIR, WORK_DIR]

DEFAULT_CLEANUP_WORDS = [
    'corporation', 'corp', 'incorporated', 'inc', 'limited', 'ltd',
    'plc', 'holding', 'trust', 'nv', 'group'
]


def clean_company_name(name: str, word_list: list) -> str:
    """Clean company names by truncating at first matched keyword."""
    if not isinstance(name, str):
        return name
    # Normalize: remove non-alphanumeric (keep spaces)
    name_lower = name.lower()
    best_pos = len(name)
    for word in word_list:
        # Build pattern: word boundary match ignoring non-alphanumeric
        pattern = re.compile(r'(?i)\b' + re.escape(word) + r'\b')
        # Search in the cleaned version but cut from original
        match = pattern.search(name_lower)
        if match:
            # Find position of match end in original string
            end_pos = match.end()
            if end_pos < best_pos:
                best_pos = end_pos
    result = name[:best_pos].strip()
    # Clean trailing punctuation
    result = re.sub(r'[\s,.\-/\\]+$', '', result)
    return result if result else name


@st.cache_data(ttl=3600)
def load_data():
    """Load and prepare all data files."""
    def resolve_input_file(label: str, exact_names: list, prefixes: list, exts: list):
        """Resolve input file from script/cwd with exact names first, then prefix scan."""
        for base_dir in SEARCH_DIRS:
            for name in exact_names:
                p = base_dir / name
                if p.exists():
                    return p
            for pref in prefixes:
                for ext in exts:
                    for p in sorted(base_dir.glob(f"{pref}*{ext}")):
                        if p.is_file():
                            return p
        searched = ", ".join(str(d) for d in SEARCH_DIRS)
        raise FileNotFoundError(
            f"Missing {label} file. Searched in: {searched}. "
            f"Tried exact names: {exact_names} and prefixes: {prefixes}"
        )

    # 1. Market cap data
    mc_path = resolve_input_file(
        label="market cap",
        exact_names=["NK_market_cap.xlsx", "NK_market_cap_20260211.xlsx"],
        prefixes=["NK_market_cap"],
        exts=[".xlsx", ".xls", ".csv"],
    )
    if mc_path.suffix.lower() == ".csv":
        df_mc = pd.read_csv(mc_path)
    else:
        df_mc = pd.read_excel(mc_path, sheet_name=0)
    df_mc.columns = df_mc.columns.str.strip()
    
    # Rename 'Name' → 'Company' if present
    if 'Name' in df_mc.columns:
        df_mc.rename(columns={'Name': 'Company'}, inplace=True)
    
    # Clean company names
    df_mc = df_mc.dropna(subset=['Symbol']).copy()
    df_mc['Symbol'] = df_mc['Symbol'].astype(str).str.strip()
    df_mc['Company'] = df_mc['Company'].apply(
        lambda x: clean_company_name(x, DEFAULT_CLEANUP_WORDS)
    )
    
    # Parse Market Cap - handle string formats like "1.5B", "500M" etc.
    def parse_market_cap(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val).strip().upper().replace('$', '').replace(',', '')
        try:
            if val_str.endswith('T'):
                return float(val_str[:-1]) * 1e12
            elif val_str.endswith('B'):
                return float(val_str[:-1]) * 1e9
            elif val_str.endswith('M'):
                return float(val_str[:-1]) * 1e6
            elif val_str.endswith('K'):
                return float(val_str[:-1]) * 1e3
            else:
                return float(val_str)
        except ValueError:
            return np.nan
    
    if 'Market Cap' in df_mc.columns:
        df_mc['Market Cap $'] = df_mc['Market Cap'].apply(parse_market_cap)
    elif 'Market Cap $' not in df_mc.columns:
        # Try to find any market cap column
        mc_cols = [c for c in df_mc.columns if 'market' in c.lower() and 'cap' in c.lower()]
        if mc_cols:
            df_mc['Market Cap $'] = df_mc[mc_cols[0]].apply(parse_market_cap)
    
    # 2. Stock price data
    stock_path = resolve_input_file(
        label="stock price",
        exact_names=["NK_stock_data.csv", "NK_stock_data_20260211.csv"],
        prefixes=["NK_stock_data"],
        exts=[".csv", ".xlsx", ".xls"],
    )
    if stock_path.suffix.lower() == ".csv":
        df_stock = pd.read_csv(stock_path, parse_dates=['Date'])
    else:
        df_stock = pd.read_excel(stock_path, parse_dates=['Date'])
    df_stock.columns = df_stock.columns.str.strip()
    df_stock = df_stock.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    
    # 3. NASDAQ Index data
    idx_path = resolve_input_file(
        label="nasdaq index",
        exact_names=["NK_nsdq_index.csv", "NK_nsdq_index_20260211.csv", "NK_nsdq_index"],
        prefixes=["NK_nsdq_index"],
        exts=[".csv", ".xlsx", ".xls", ""],
    )
    if idx_path.suffix.lower() in [".xlsx", ".xls"]:
        df_idx = pd.read_excel(idx_path, parse_dates=['Date'])
    else:
        df_idx = pd.read_csv(idx_path, parse_dates=['Date'])
    
    df_idx.columns = df_idx.columns.str.strip()
    df_idx = df_idx.sort_values('Date').reset_index(drop=True)
    
    # Merge stock + market cap
    df_merged = df_stock.merge(df_mc[['Symbol', 'Company', 'Market Cap $', 'Sector', 'Industry']],
                                on='Symbol', how='left')
    
    return df_mc, df_stock, df_idx, df_merged


try:
    df_mc, df_stock, df_idx, df_merged = load_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    load_error = str(e)


# ─── Utility Functions ───────────────────────────────────────────────────────

def format_number(n, decimals=0):
    """Format large numbers with K/M/B suffixes."""
    if pd.isna(n):
        return "N/A"
    if abs(n) >= 1e12:
        return f"${n/1e12:,.{decimals}f}T"
    elif abs(n) >= 1e9:
        return f"${n/1e9:,.{decimals}f}B"
    elif abs(n) >= 1e6:
        return f"${n/1e6:,.{decimals}f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:,.{decimals}f}K"
    else:
        return f"{n:,.{decimals}f}"


def format_count(n):
    """Format counts with K suffix."""
    if n >= 1000:
        return f"{n/1000:,.1f}K"
    return f"{n:,}"


def with_row_ref(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 1-based Row Ref column for display."""
    out = df.copy().reset_index(drop=True)
    out.insert(0, 'Row Ref', np.arange(1, len(out) + 1))
    return out


def compute_year_bound_prices(df_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute yearly start/end/return for each symbol."""
    if df_prices.empty:
        return pd.DataFrame(columns=['Symbol', 'Year', 'Start Price', 'End Price', 'Stock Return'])
    tmp = df_prices.copy()
    tmp['Year'] = tmp['Date'].dt.year
    yearly = tmp.sort_values(['Symbol', 'Date']).groupby(['Symbol', 'Year']).agg(
        Start_Price=('Close', 'first'),
        End_Price=('Close', 'last')
    ).reset_index()
    yearly['Stock_Return'] = (yearly['End_Price'] / yearly['Start_Price'] - 1) * 100
    yearly.rename(columns={
        'Start_Price': 'Start Price',
        'End_Price': 'End Price',
        'Stock_Return': 'Stock Return'
    }, inplace=True)
    return yearly


def compute_index_year_returns(df_index: pd.DataFrame) -> pd.DataFrame:
    """Compute yearly NASDAQ returns."""
    idx = df_index.copy().sort_values('Date')
    idx['Year'] = idx['Date'].dt.year
    out = idx.groupby('Year').agg(
        Start=('Close', 'first'),
        End=('Close', 'last')
    ).reset_index()
    out['Nasdaq Return'] = (out['End'] / out['Start'] - 1) * 100
    return out[['Year', 'Nasdaq Return']]


def percentile_52w(df_symbol: pd.DataFrame) -> float:
    """Current price percentile over latest 52 weeks (~252 trading days)."""
    if df_symbol.empty:
        return np.nan
    s = df_symbol.sort_values('Date')['Close']
    latest = s.iloc[-1]
    window = s.tail(252)
    return (window <= latest).mean() * 100 if len(window) > 0 else np.nan


def compute_streaks(df_symbol):
    """
    Compute consecutive up/down streaks for a single stock's price history.
    Returns the dataframe with additional 'daily_return', 'direction', 'streak' columns.
    """
    df = df_symbol.copy().sort_values('Date').reset_index(drop=True)
    df['daily_return'] = df['Close'].pct_change()
    df['direction'] = np.where(df['daily_return'] > 0, 1, np.where(df['daily_return'] < 0, -1, 0))
    
    # Compute streak length
    streaks = []
    current_streak = 0
    current_dir = 0
    for i, row in df.iterrows():
        d = row['direction']
        if d == current_dir and d != 0:
            current_streak += 1
        elif d != 0:
            current_dir = d
            current_streak = 1
        else:
            current_streak = 0
            current_dir = 0
        streaks.append(current_streak * current_dir)
    df['streak'] = streaks
    return df


def find_streak_events(df_with_streaks, streak_length, direction='up'):
    """
    Find dates where a streak of exactly `streak_length` consecutive
    up (direction='up') or down (direction='down') days ends.
    Returns list of (date, streak_end_idx) tuples.
    """
    target = streak_length if direction == 'up' else -streak_length
    events = []
    
    abs_streaks = df_with_streaks['streak'].values
    for i in range(len(abs_streaks)):
        if direction == 'up' and abs_streaks[i] >= target:
            # Check if next day breaks the streak (or it's the last day)
            if i == len(abs_streaks) - 1 or abs_streaks[i + 1] < abs_streaks[i] + 1:
                if abs_streaks[i] >= target:
                    events.append(i)
        elif direction == 'down' and abs_streaks[i] <= target:
            if i == len(abs_streaks) - 1 or abs_streaks[i + 1] > abs_streaks[i] - 1:
                if abs_streaks[i] <= target:
                    events.append(i)
    
    return events


def compute_forward_returns(df_stock_sorted, event_indices, return_window, df_idx_sorted):
    """
    Compute forward returns for stock and index after each event.
    """
    results = []
    stock_dates = df_stock_sorted['Date'].values
    stock_closes = df_stock_sorted['Close'].values
    
    idx_dates = df_idx_sorted['Date'].values
    idx_closes = df_idx_sorted['Close'].values
    
    for idx in event_indices:
        event_date = stock_dates[idx]
        event_price = stock_closes[idx]
        
        # Find forward price after return_window trading days
        future_idx = idx + return_window
        if future_idx >= len(stock_closes):
            continue
        
        future_price = stock_closes[future_idx]
        stock_return = (future_price / event_price - 1) * 100
        
        # Find matching index return
        event_date_ts = pd.Timestamp(event_date)
        future_date_ts = pd.Timestamp(stock_dates[future_idx])
        
        # Find nearest index dates
        idx_event_mask = idx_dates <= np.datetime64(event_date_ts)
        idx_future_mask = idx_dates <= np.datetime64(future_date_ts)
        
        if idx_event_mask.any() and idx_future_mask.any():
            idx_event_price = idx_closes[np.where(idx_event_mask)[0][-1]]
            idx_future_price = idx_closes[np.where(idx_future_mask)[0][-1]]
            nasdaq_return = (idx_future_price / idx_event_price - 1) * 100
        else:
            nasdaq_return = np.nan
        
        results.append({
            'event_date': pd.Timestamp(event_date),
            'event_price': event_price,
            'future_date': pd.Timestamp(stock_dates[future_idx]),
            'future_price': future_price,
            'stock_return': stock_return,
            'nasdaq_return': nasdaq_return,
        })
    
    return results


def get_time_filter_date(option, max_date):
    """Return start date based on time period selection."""
    if option == 'All data period available':
        return None
    mapping = {
        'Past 5 years': 365 * 5,
        'Past 2 years': 365 * 2,
        'Past 1 year': 365,
        'Past 6 months': 183,
        'Past 3 months': 91,
    }
    days = mapping.get(option, 0)
    if days:
        return max_date - timedelta(days=days)
    return None


# ============================================================================
# FORMATTING FUNCTIONS | DAILY CLARION POSITION - TABLES
# ============================================================================

# RED = "#C00000"

def format_value(val):
    """
    Formats a numeric value for Streamlit HTML display.
    Returns an HTML string with appropriate formatting and color.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"

    is_negative = val < 0
    abs_val = abs(val)

    if val == 0 or pd.isna(val):
        return "-"
    # Rule 1: Basis points — value is a percentage, abs < 0.99%
    if abs_val < 0.0099:
        bps = abs_val * 10000  # convert percentage to bps
        formatted = f"({bps:.2f})Bps" if is_negative else f"{bps:.2f} Bps"
        # color = f' style="color:{RED}"' if is_negative else ""
        # return f'<span{color}>{formatted}</span>'
        return formatted

    # Rule 2: Has decimal fraction
    if val != int(val):
        if is_negative:
            return f"({abs_val:,.2f})"
        return f"{val:,.2f}"

    # Rule 3: Integer
    if is_negative:
        return f"({abs_val:,.0f})"
    return f"{val:,.0f}"


def format_dataframe(df: pd.DataFrame, numeric_cols: list = None) -> pd.DataFrame:
    """
    Applies format_value to all (or specified) numeric columns.
    Returns a new DataFrame with HTML-formatted string values.
    """
    df_fmt = df.copy()
    cols = numeric_cols or df.select_dtypes(include="number").columns.tolist()
    for col in cols:
        df_fmt[col] = df_fmt[col].apply(format_value)
    return df_fmt

def add_header_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an Excel-style column reference row [A], [B], [C]...
    Works safely even if df already has MultiIndex columns.
    """

    # If already MultiIndex → flatten first
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[-1]) for col in df.columns]

    n = len(df.columns)

    # Generate A, B, C, ..., Z, AA, AB if needed
    def excel_letters(n):
        letters = []
        i = 0
        while len(letters) < n:
            s = ""
            x = i
            while True:
                s = chr(65 + x % 26) + s
                x = x // 26 - 1
                if x < 0:
                    break
            letters.append(f"[{s}]")
            i += 1
        return letters

    head_idx = excel_letters(n)

    # Ensure same length
    if len(head_idx) != n:
        raise ValueError(
            f"Header index mismatch: {len(head_idx)} vs {n} columns"
        )

    df = df.copy()
    df.columns = pd.MultiIndex.from_arrays(
        [head_idx, df.columns.astype(str)]
    )

    return df


def center_top_index(styler):
    """
    Center-align the first (top) column index level.
    """
    return styler.set_table_styles([
        {
            "selector": "th.col_heading.level0",
            "props": [("text-align", "center")]
        }
    ])

# ─── Main App ────────────────────────────────────────────────────────────────

st.markdown("# 📈 NASDAQ Stock Streak Screener")
st.markdown("*Screen NASDAQ stocks for consecutive price streaks and analyze forward returns vs. the NASDAQ Composite Index*")

if not data_loaded:
    st.error(f"❌ **Failed to load data:** {load_error}")
    st.info("Ensure all data files are in the expected directory and file names match.")
    st.stop()

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 [1] Stock Price History",
    "🔍 [2] Stock Returns Streak Analysis",
    "📉 [3] Macro Level Analysis",
    "🗓️ [4] Annual Returns Analysis"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Stock Price History
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Stock Price Explorer</div>', unsafe_allow_html=True)
    
    # Build symbol list: stocks + index
    stock_symbols = sorted(df_mc['Symbol'].dropna().unique().tolist())
    all_symbols = stock_symbols + ['^IXIC (NASDAQ Composite)']
    
    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        selected_symbol = st.selectbox("Select Stock / Index", all_symbols, index=0, key='tab1_symbol')
    
    is_index = selected_symbol.startswith('^IXIC')
    symbol_key = '^IXIC' if is_index else selected_symbol
    
    if is_index:
        df_sel = df_idx[df_idx['Symbol'] == '^IXIC'].copy()
        company_name = "NASDAQ Composite Index"
        sector_val = "Index"
        industry_val = "Index"
        mcap_val = "N/A"
    else:
        df_sel = df_stock[df_stock['Symbol'] == symbol_key].copy()
        mc_row = df_mc[df_mc['Symbol'] == symbol_key]
        company_name = mc_row['Company'].values[0] if len(mc_row) > 0 else symbol_key
        sector_val = mc_row['Sector'].values[0] if len(mc_row) > 0 else "N/A"
        industry_val = mc_row['Industry'].values[0] if len(mc_row) > 0 else "N/A"
        mcap_val = format_number(mc_row['Market Cap $'].values[0], 1) if len(mc_row) > 0 else "N/A"
    
    if len(df_sel) == 0:
        st.warning(f"No price data available for {symbol_key}")
    else:
        df_sel = df_sel.sort_values('Date').reset_index(drop=True)
        
        min_price = df_sel['Close'].min()
        max_price = df_sel['Close'].max()
        min_date = df_sel.loc[df_sel['Close'].idxmin(), 'Date']
        max_date = df_sel.loc[df_sel['Close'].idxmax(), 'Date']
        latest_price = df_sel['Close'].iloc[-1]
        latest_date = df_sel['Date'].iloc[-1]
        first_date = df_sel['Date'].iloc[0]
        percentile = (df_sel['Close'] <= latest_price).mean() * 100
        
        # Stock Details Card
        st.markdown(f"### {company_name} (`{symbol_key}`)")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sector", sector_val)
        c2.metric("Industry", industry_val)
        c3.metric("Market Cap", mcap_val)
        c4.metric("Latest Price", f"${latest_price:,.2f}")
        c5.metric("Price Percentile", f"{percentile:.1f}%")
        
        c6, c7, c8, c9 = st.columns(4)
        c6.metric("Date Range", f"{first_date.strftime('%Y-%m-%d %A')} → {latest_date.strftime('%Y-%m-%d %A')}")
        c7.metric("Trading Days", f"{len(df_sel):,}")
        c8.metric("Min Price", f"${min_price:,.2f} ({min_date.strftime('%Y-%m-%d %A')})")
        c9.metric("Max Price", f"${max_price:,.2f} ({max_date.strftime('%Y-%m-%d %A')})")
        
        # Price History Table
        st.markdown('<div class="section-header">Closing Price History</div>', unsafe_allow_html=True)
        df_display = df_sel[['Date', 'Close']].copy()
        df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d %A')
        df_display = df_display.sort_values('Date', ascending=False)
        df_display = with_row_ref(df_display)
        df_display = format_dataframe(df_display)
        df_display = add_header_index(df_display)
        df_styled = df_display.style.pipe(center_top_index)
        st.dataframe(df_styled, use_container_width=True, height=300, hide_index=True)
        
        # Chart: Stock vs NASDAQ Composite
        st.markdown('<div class="section-header">Price Chart vs NASDAQ Composite</div>', unsafe_allow_html=True)
        
        if not is_index:
            # Merge stock and index on overlapping dates
            df_chart_stock = df_sel[['Date', 'Close']].rename(columns={'Close': 'Stock Price'})
            df_chart_idx = df_idx[['Date', 'Close']].rename(columns={'Close': 'NASDAQ Composite'})
            df_chart = df_chart_stock.merge(df_chart_idx, on='Date', how='inner')
            
            # Normalize to 100 at start
            df_chart['Stock (Indexed)'] = df_chart['Stock Price'] / df_chart['Stock Price'].iloc[0] * 100
            df_chart['NASDAQ (Indexed)'] = df_chart['NASDAQ Composite'] / df_chart['NASDAQ Composite'].iloc[0] * 100
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=df_chart['Date'], y=df_chart['Stock Price'],
                          name=f"{symbol_key} Price", line=dict(color='#00d4ff', width=2)),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=df_chart['Date'], y=df_chart['NASDAQ Composite'],
                          name="NASDAQ Composite", line=dict(color='#ff6b6b', width=1.5, dash='dot')),
                secondary_y=True
            )
            fig.update_layout(
                template='plotly_dark',
                height=500,
                title=f"{company_name} vs NASDAQ Composite",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=60, r=60, t=80, b=40),
            )
            fig.update_yaxes(title_text=f"{symbol_key} ($)", secondary_y=False)
            fig.update_yaxes(title_text="NASDAQ Composite", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Indexed comparison
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['Stock (Indexed)'],
                                      name=symbol_key, line=dict(color='#00d4ff', width=2)))
            fig2.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['NASDAQ (Indexed)'],
                                      name="NASDAQ Composite", line=dict(color='#ff6b6b', width=2)))
            fig2.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
            fig2.update_layout(
                template='plotly_dark', height=400,
                title="Indexed Performance (Base = 100)",
                yaxis_title="Indexed Value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=60, r=60, t=80, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_sel['Date'], y=df_sel['Close'],
                                     name="NASDAQ Composite", line=dict(color='#00d4ff', width=2),
                                     fill='tozeroy', fillcolor='rgba(0,212,255,0.1)'))
            fig.update_layout(
                template='plotly_dark', height=500,
                title="NASDAQ Composite Index",
                yaxis_title="Index Value",
                margin=dict(l=60, r=60, t=80, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Stock Returns Streak Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    
    # ── Section 1: Population Summary ────────────────────────────────────────
    st.markdown('<div class="section-header">Section 1 — Stock Population Summary</div>', unsafe_allow_html=True)
    
    total_stocks = df_mc['Symbol'].nunique()
    date_min = df_stock['Date'].min()
    date_max = df_stock['Date'].max()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Stocks", format_count(total_stocks))
    m2.metric("Date Range Start", date_min.strftime('%Y-%m-%d %A'))
    m3.metric("Date Range End", date_max.strftime('%Y-%m-%d %A'))
    m4.metric("Total Trading Days", f"{df_stock['Date'].nunique():,}")
    
    pop_col1, pop_col2 = st.columns(2)
    
    with pop_col1:
        # Market Cap distribution
        st.markdown("**Market Cap Distribution**")
        df_mc_valid = df_mc.dropna(subset=['Market Cap $'])
        bins = [0, 1e9, 5e9, 20e9, 50e9, 200e9, np.inf]
        labels = ['< $1B', '$1B - $5B', '$5B - $20B', '$20B - $50B', '$50B - $200B', '> $200B']
        df_mc_valid = df_mc_valid.copy()
        df_mc_valid['MC Bucket'] = pd.cut(df_mc_valid['Market Cap $'], bins=bins, labels=labels, right=False)
        mc_dist = df_mc_valid['MC Bucket'].value_counts().sort_index().reset_index()
        mc_dist.columns = ['Market Cap Range', 'Count']
        mc_dist = with_row_ref(mc_dist)
        mc_dist = format_dataframe(mc_dist)
        mc_dist = add_header_index(mc_dist)
        mc_dist_styled = mc_dist.style.pipe(center_top_index)
        st.dataframe(mc_dist_styled, use_container_width=True, hide_index=True)
    
    with pop_col2:
        # Sector distribution
        st.markdown("**Sector Distribution**")
        sector_dist = df_mc['Sector'].value_counts().reset_index()
        sector_dist.columns = ['Sector', 'Count']
        sector_dist = with_row_ref(sector_dist)
        sector_dist = format_dataframe(sector_dist)
        sector_dist = add_header_index(sector_dist)
        sector_dist_styled = sector_dist.style.pipe(center_top_index)
        st.dataframe(sector_dist_styled, use_container_width=True, hide_index=True)
    
    # Industry distribution (collapsible)
    with st.expander("Industry Distribution", expanded=False):
        ind_dist = df_mc['Industry'].value_counts().reset_index()
        ind_dist.columns = ['Industry', 'Count']
        st.dataframe(with_row_ref(ind_dist), use_container_width=True, hide_index=True, height=400)
    
    st.divider()
    
    # ── Section 2: Exclusion Filters ─────────────────────────────────────────
    st.markdown('<div class="section-header">Section 2 — Exclusion Filters</div>', unsafe_allow_html=True)
    
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        mcap_filter = st.selectbox(
            "A. Min Market Cap (exclude below)",
            ['No filter', '$5 Billion', '$20 Billion', '$50 Billion'],
            index=1, key='mcap_filter'
        )
        mcap_threshold = {
            'No filter': 0,
            '$5 Billion': 5e9,
            '$20 Billion': 20e9,
            '$50 Billion': 50e9,
        }[mcap_filter]
    
    with fc2:
        all_sectors = sorted(df_mc['Sector'].dropna().unique().tolist())
        exclude_sectors = st.multiselect("B. Exclude Sectors", all_sectors, default=[], key='excl_sectors')
    
    with fc3:
        all_industries = sorted(df_mc['Industry'].dropna().unique().tolist())
        exclude_industries = st.multiselect("C. Exclude Industries", all_industries, default=[], key='excl_industries')
    
    fc4, fc5, fc6 = st.columns(3)
    with fc4:
        time_period = st.selectbox(
            "D. Time Period",
            ['All data period available', 'Past 5 years', 'Past 2 years', 'Past 1 year', 'Past 6 months', 'Past 3 months'],
            index=0, key='time_period'
        )
    with fc5:
        event_type = st.selectbox(
            "E. Consecutive Day Event",
            ['3 days', '5 days', '7 days', '10 days'],
            index=0, key='event_type'
        )
        streak_len = int(event_type.split()[0])
    with fc6:
        event_direction = st.selectbox(
            "Event Direction",
            ['Both', 'Increase (UP)', 'Decrease (DOWN)'],
            index=0, key='event_direction'
        )
        direction_map = {
            'Both': 'both',
            'Increase (UP)': 'up',
            'Decrease (DOWN)': 'down',
        }
        selected_direction = direction_map[event_direction]

    fc7, _, _ = st.columns(3)
    with fc7:
        return_window = st.selectbox(
            "F. Return Comparison Window",
            ['30 days', '60 days', '90 days', '120 days'],
            index=1, key='return_window'
        )
        return_days = int(return_window.split()[0])
    
    # Apply filters
    df_filtered_mc = df_mc.copy()
    
    # Track waterfall
    waterfall_steps = [{'Stage': 'Total Stocks', 'Count': len(df_filtered_mc), 'Excluded': 0}]
    
    # A: Market Cap filter
    if mcap_threshold > 0:
        before = len(df_filtered_mc)
        df_filtered_mc = df_filtered_mc[df_filtered_mc['Market Cap $'] >= mcap_threshold]
        excluded = before - len(df_filtered_mc)
        waterfall_steps.append({
            'Stage': f'Market Cap ≥ {mcap_filter}',
            'Count': len(df_filtered_mc),
            'Excluded': excluded
        })
    
    # B: Sector filter
    if exclude_sectors:
        before = len(df_filtered_mc)
        df_filtered_mc = df_filtered_mc[~df_filtered_mc['Sector'].isin(exclude_sectors)]
        excluded = before - len(df_filtered_mc)
        waterfall_steps.append({
            'Stage': f'Excl. {len(exclude_sectors)} Sector(s)',
            'Count': len(df_filtered_mc),
            'Excluded': excluded
        })
    
    # C: Industry filter
    if exclude_industries:
        before = len(df_filtered_mc)
        df_filtered_mc = df_filtered_mc[~df_filtered_mc['Industry'].isin(exclude_industries)]
        excluded = before - len(df_filtered_mc)
        waterfall_steps.append({
            'Stage': f'Excl. {len(exclude_industries)} Industry(ies)',
            'Count': len(df_filtered_mc),
            'Excluded': excluded
        })
    
    # Check stocks have price data
    symbols_with_prices = set(df_stock['Symbol'].unique())
    before = len(df_filtered_mc)
    df_filtered_mc = df_filtered_mc[df_filtered_mc['Symbol'].isin(symbols_with_prices)]
    excluded = before - len(df_filtered_mc)
    if excluded > 0:
        waterfall_steps.append({
            'Stage': 'Has Price Data',
            'Count': len(df_filtered_mc),
            'Excluded': excluded
        })
    
    in_scope_symbols = df_filtered_mc['Symbol'].unique().tolist()
    
    # Time filter on price data
    time_start = get_time_filter_date(time_period, date_max)
    
    st.divider()
    
    # ── Section 3: Waterfall ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Section 3 — Waterfall: Stocks in Scope</div>', unsafe_allow_html=True)
    
    wf = pd.DataFrame(waterfall_steps)
    
    # Build waterfall chart
    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * (len(wf) - 1),
        x=wf['Stage'].tolist(),
        y=[wf['Count'].iloc[0]] + [-wf['Excluded'].iloc[i] for i in range(1, len(wf))],
        textposition="outside",
        text=[f"{format_count(wf['Count'].iloc[0])}"] + [f"-{format_count(wf['Excluded'].iloc[i])}" for i in range(1, len(wf))],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#0f3460"}},
        decreasing={"marker": {"color": "#ff6b6b"}},
        totals={"marker": {"color": "#00d4ff"}},
    ))
    fig_wf.update_layout(
        template='plotly_dark', height=400,
        title=f"Stock Universe Filtering — {format_count(len(in_scope_symbols))} stocks in scope",
        showlegend=False,
        margin=dict(l=60, r=60, t=80, b=40),
    )
    st.plotly_chart(fig_wf, use_container_width=True)
    
    st.info(f"**{format_count(len(in_scope_symbols))} stocks** remain after applying all exclusion filters.")
    
    st.divider()
    
    # ── Streak Computation (cached per filter set) ───────────────────────────
    # We'll compute streaks for in-scope stocks
    
    @st.cache_data(ttl=600)
    def compute_all_streaks_and_returns(_symbols, _streak_len, _return_days, _time_start_str, _df_stock_hash):
        """Compute streak events and returns for all symbols."""
        
        time_start_dt = pd.Timestamp(_time_start_str) if _time_start_str else None
        
        all_events_up = []
        all_events_down = []
        recent_up = []
        recent_down = []
        
        cutoff_14d = date_max - timedelta(days=14)
        
        idx_sorted = df_idx.sort_values('Date').reset_index(drop=True)
        
        for sym in _symbols:
            df_sym = df_stock[df_stock['Symbol'] == sym].copy()
            if len(df_sym) < _streak_len + _return_days:
                continue
            
            df_sym = df_sym.sort_values('Date').reset_index(drop=True)
            
            # Apply time filter
            if time_start_dt:
                df_sym_filtered = df_sym[df_sym['Date'] >= time_start_dt].reset_index(drop=True)
            else:
                df_sym_filtered = df_sym.copy()
            
            if len(df_sym_filtered) < _streak_len:
                continue
            
            # Compute streaks
            df_streaked = compute_streaks(df_sym_filtered)
            
            # Find UP events
            up_indices = find_streak_events(df_streaked, _streak_len, 'up')
            if up_indices:
                returns = compute_forward_returns(df_streaked, up_indices, _return_days, idx_sorted)
                for r in returns:
                    r['Symbol'] = sym
                    r['direction'] = 'up'
                all_events_up.extend(returns)
                
                # Check recent events (last 14 days)
                for idx_val in up_indices:
                    if pd.Timestamp(df_streaked.loc[idx_val, 'Date']) >= cutoff_14d:
                        recent_up.append({
                            'Symbol': sym,
                            'event_date': df_streaked.loc[idx_val, 'Date'],
                            'price_at_event': df_streaked.loc[idx_val, 'Close'],
                        })
            
            # Find DOWN events
            down_indices = find_streak_events(df_streaked, _streak_len, 'down')
            if down_indices:
                returns = compute_forward_returns(df_streaked, down_indices, _return_days, idx_sorted)
                for r in returns:
                    r['Symbol'] = sym
                    r['direction'] = 'down'
                all_events_down.extend(returns)
                
                # Check recent events
                for idx_val in down_indices:
                    if pd.Timestamp(df_streaked.loc[idx_val, 'Date']) >= cutoff_14d:
                        recent_down.append({
                            'Symbol': sym,
                            'event_date': df_streaked.loc[idx_val, 'Date'],
                            'price_at_event': df_streaked.loc[idx_val, 'Close'],
                        })
        
        return all_events_up, all_events_down, recent_up, recent_down
    
    if len(in_scope_symbols) > 0:
        with st.spinner(f"Computing {streak_len}-day streaks for {len(in_scope_symbols)} stocks..."):
            time_start_str = str(time_start) if time_start else None
            stock_hash = f"{len(df_stock)}_{df_stock['Date'].max()}"
            
            all_events_up, all_events_down, recent_up, recent_down = compute_all_streaks_and_returns(
                tuple(in_scope_symbols), streak_len, return_days, time_start_str, stock_hash
            )
        
        # ── Section 4: Recent Opportunity Windows ────────────────────────────
        st.markdown('<div class="section-header">Section 4 — Recent Opportunity Windows (Last 14 Days)</div>', unsafe_allow_html=True)
        
        def build_recent_table(recent_events, direction_label):
            if not recent_events:
                return pd.DataFrame()
            df_recent = pd.DataFrame(recent_events)
            df_recent = df_recent.merge(
                df_filtered_mc[['Symbol', 'Company', 'Sector', 'Industry']],
                on='Symbol', how='left'
            )
            df_recent.dropna(subset=['Company'], inplace=True)
            
            # Add min/max/percentile from time-filtered data
            enriched = []
            for _, row in df_recent.iterrows():
                sym = row['Symbol']
                df_sym = df_stock[df_stock['Symbol'] == sym]
                if time_start:
                    df_sym = df_sym[df_sym['Date'] >= time_start]
                if len(df_sym) > 0:
                    min_p = df_sym['Close'].min()
                    max_p = df_sym['Close'].max()
                    latest = df_sym.sort_values('Date')['Close'].iloc[-1]
                    pctl = (df_sym['Close'] <= latest).mean() * 100
                else:
                    min_p = max_p = latest = pctl = np.nan
                enriched.append({
                    'Symbol': sym,
                    'Company': row.get('Company', ''),
                    'Sector': row.get('Sector', ''),
                    'Industry': row.get('Industry', ''),
                    'Most Recent Price': f"${latest:,.2f}" if not pd.isna(latest) else 'N/A',
                    'Min Price (Period)': f"${min_p:,.2f}" if not pd.isna(min_p) else 'N/A',
                    'Max Price (Period)': f"${max_p:,.2f}" if not pd.isna(max_p) else 'N/A',
                    'Price Percentile': f"{pctl:.1f}%" if not pd.isna(pctl) else 'N/A',
                })
            return pd.DataFrame(enriched)
        
        show_up = selected_direction in ['both', 'up']
        show_down = selected_direction in ['both', 'down']
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            if show_up:
                st.markdown(f"**🟢 {streak_len}-Day UP Streaks (Last 14 Days)**")
                df_recent_up = build_recent_table(recent_up, 'UP')
                df_recent_up = with_row_ref(df_recent_up)
                df_recent_up = format_dataframe(df_recent_up)
                df_recent_up = add_header_index(df_recent_up)
                df_recent_up_styled = df_recent_up.style.pipe(center_top_index)
                if len(df_recent_up) > 0:
                    st.dataframe(df_recent_up_styled, use_container_width=True, hide_index=True)
                else:
                    st.info("No UP streak events in the last 14 days.")
        with r_col2:
            if show_down:
                st.markdown(f"**🔴 {streak_len}-Day DOWN Streaks (Last 14 Days)**")
                df_recent_down = build_recent_table(recent_down, 'DOWN')
                df_recent_down = with_row_ref(df_recent_down)
                df_recent_down = format_dataframe(df_recent_down)
                df_recent_down = add_header_index(df_recent_down)
                df_recent_down_styled = df_recent_down.style.pipe(center_top_index)
                if len(df_recent_down) > 0:
                    st.dataframe(df_recent_down_styled, use_container_width=True, hide_index=True)
                else:
                    st.info("No DOWN streak events in the last 14 days.")
        
        st.divider()
        
        # ── Section 5: Summary Table ─────────────────────────────────────────
        st.markdown('<div class="section-header">Section 5 — Streak Analysis Summary</div>', unsafe_allow_html=True)
        
        def build_summary(events_list, direction_label):
            if not events_list:
                return pd.DataFrame()
            df_ev = pd.DataFrame(events_list)
            
            summary = df_ev.groupby('Symbol').agg(
                num_events=('stock_return', 'count'),
                earliest=('event_date', 'min'),
                latest=('event_date', 'max'),
                avg_stock_return=('stock_return', 'mean'),
                max_stock_return=('stock_return', 'max'),
                min_stock_return=('stock_return', 'min'),
                avg_nasdaq_return=('nasdaq_return', 'mean'),
                max_nasdaq_return=('nasdaq_return', 'max'),
                min_nasdaq_return=('nasdaq_return', 'min'),
            ).reset_index()
            
            # Merge company info
            summary = summary.merge(
                df_filtered_mc[['Symbol', 'Company', 'Market Cap $', 'Sector', 'Industry']],
                on='Symbol', how='left'
            )
            
            # Format
            summary['Market Cap $'] = summary['Market Cap $'].apply(lambda x: format_number(x, 1))
            summary['earliest'] = summary['earliest'].dt.strftime('%Y-%m-%d %A')
            summary['latest'] = summary['latest'].dt.strftime('%Y-%m-%d %A')
            
            for col in ['avg_stock_return', 'max_stock_return', 'min_stock_return',
                        'avg_nasdaq_return', 'max_nasdaq_return', 'min_nasdaq_return']:
                summary[col] = summary[col].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else 'N/A')
            
            summary = summary.rename(columns={
                'num_events': '# Events',
                'earliest': 'Earliest',
                'latest': 'Latest',
                'avg_stock_return': 'Avg Stock Return',
                'max_stock_return': 'Max Stock Return',
                'min_stock_return': 'Min Stock Return',
                'avg_nasdaq_return': 'Avg NASDAQ Return',
                'max_nasdaq_return': 'Max NASDAQ Return',
                'min_nasdaq_return': 'Min NASDAQ Return',
            })
            
            cols_order = ['Company', 'Symbol', 'Market Cap $', 'Sector', 'Industry',
                         '# Events', 'Earliest', 'Latest',
                         'Avg Stock Return', 'Max Stock Return', 'Min Stock Return',
                         'Avg NASDAQ Return', 'Max NASDAQ Return', 'Min NASDAQ Return']
            return summary[[c for c in cols_order if c in summary.columns]].sort_values('# Events', ascending=False)
        
        # UP streaks summary
        if show_up:
            st.markdown(f"**🟢 {streak_len}-Day Consecutive UP Streaks → {return_days}-Day Forward Returns**")
            df_summary_up = build_summary(all_events_up, 'UP')
            df_summary_up_display = with_row_ref(df_summary_up)
            df_summary_up_display = format_dataframe(df_summary_up_display)
            df_summary_up_display = add_header_index(df_summary_up_display)
            df_summary_up_styled = df_summary_up_display.style.pipe(center_top_index)
            if len(df_summary_up) > 0:
                st.metric("Total UP Events", f"{len(all_events_up):,} across {df_summary_up['Symbol'].nunique()} stocks")
                st.dataframe(df_summary_up_styled, use_container_width=True, hide_index=True, height=400)
            else:
                st.info("No UP streak events found with current filters.")
        
        # DOWN streaks summary
        if show_down:
            st.markdown(f"**🔴 {streak_len}-Day Consecutive DOWN Streaks → {return_days}-Day Forward Returns**")
            df_summary_down = build_summary(all_events_down, 'DOWN')
            df_summary_down_display = with_row_ref(df_summary_down)
            df_summary_down_display = format_dataframe(df_summary_down_display)
            df_summary_down_display = add_header_index(df_summary_down_display)
            df_summary_down_styled = df_summary_down_display.style.pipe(center_top_index)
            if len(df_summary_down) > 0:
                st.metric("Total DOWN Events", f"{len(all_events_down):,} across {df_summary_down['Symbol'].nunique()} stocks")
                st.dataframe(df_summary_down_styled, use_container_width=True, hide_index=True, height=400)
            else:
                st.info("No DOWN streak events found with current filters.")
    
    else:
        st.warning("No stocks remain after applying filters. Please adjust your exclusion criteria.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Macro Level Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Macro Level Statistics</div>', unsafe_allow_html=True)
    st.caption("Analysis based on filters applied in the **Stock Returns Streak Analysis** tab.")
    
    # Aggregation level selector
    sectors_available = sorted(df_filtered_mc['Sector'].dropna().unique().tolist())
    agg_options = ['All Stocks (Aggregate)'] + sectors_available
    agg_level = st.selectbox(
        "View Level",
        agg_options,
        index=0, key='macro_agg'
    )
    
    if len(in_scope_symbols) == 0:
        st.warning("No stocks in scope. Please adjust filters in Tab 2.")
    
    # ── Compute probability tables & return matrices ─────────────────────────
    @st.cache_data(ttl=600)
    def compute_macro_stats(_symbols, _time_start_str, _df_stock_hash):
        """Compute streak probability and return stats for all symbols."""
        time_start_dt = pd.Timestamp(_time_start_str) if _time_start_str else None
        idx_sorted = df_idx.sort_values('Date').reset_index(drop=True)
        
        all_streak_data = []  # (symbol, sector, max_up_streak, max_down_streak, streak_counts_up, streak_counts_down)
        all_return_data = []
        
        for sym in _symbols:
            df_sym = df_stock[df_stock['Symbol'] == sym].copy()
            if len(df_sym) < 15:
                continue
            df_sym = df_sym.sort_values('Date').reset_index(drop=True)
            
            if time_start_dt:
                df_sym = df_sym[df_sym['Date'] >= time_start_dt].reset_index(drop=True)
            if len(df_sym) < 15:
                continue
            
            mc_info = df_filtered_mc[df_filtered_mc['Symbol'] == sym]
            sector = mc_info['Sector'].values[0] if len(mc_info) > 0 else 'Unknown'
            
            df_s = compute_streaks(df_sym)
            
            # Count streak lengths
            for streak_day in range(1, 11):
                # UP
                up_events = find_streak_events(df_s, streak_day, 'up')
                down_events = find_streak_events(df_s, streak_day, 'down')
                
                all_streak_data.append({
                    'Symbol': sym, 'Sector': sector,
                    'streak_days': streak_day,
                    'up_count': len(up_events),
                    'down_count': len(down_events),
                    'total_days': len(df_s),
                })
                
                # Returns for key windows
                for rw in [30, 60, 90, 120]:
                    if up_events:
                        up_returns = compute_forward_returns(df_s, up_events, rw, idx_sorted)
                        for r in up_returns:
                            all_return_data.append({
                                'Symbol': sym, 'Sector': sector,
                                'streak_days': streak_day, 'direction': 'up',
                                'return_window': rw,
                                'stock_return': r['stock_return'],
                                'nasdaq_return': r['nasdaq_return'],
                            })
                    if down_events:
                        down_returns = compute_forward_returns(df_s, down_events, rw, idx_sorted)
                        for r in down_returns:
                            all_return_data.append({
                                'Symbol': sym, 'Sector': sector,
                                'streak_days': streak_day, 'direction': 'down',
                                'return_window': rw,
                                'stock_return': r['stock_return'],
                                'nasdaq_return': r['nasdaq_return'],
                            })
        
        return all_streak_data, all_return_data
    
    all_streak_data, all_return_data = [], []
    if len(in_scope_symbols) > 0:
        with st.spinner("Computing macro statistics..."):
            time_start_str2 = str(time_start) if time_start else None
            stock_hash2 = f"{len(df_stock)}_{df_stock['Date'].max()}"
            all_streak_data, all_return_data = compute_macro_stats(
                tuple(in_scope_symbols), time_start_str2, stock_hash2
            )

    if not all_streak_data:
        st.info("Insufficient data for macro analysis with current filters.")
    else:
        df_streaks_all = pd.DataFrame(all_streak_data)
        df_returns_all = pd.DataFrame(all_return_data) if all_return_data else pd.DataFrame()

        df_streaks = df_streaks_all if agg_level == 'All Stocks (Aggregate)' else df_streaks_all[df_streaks_all['Sector'] == agg_level]
        df_returns = df_returns_all if agg_level == 'All Stocks (Aggregate)' else df_returns_all[df_returns_all['Sector'] == agg_level]
        st.markdown(f"#### {'All Stocks in Scope' if agg_level == 'All Stocks (Aggregate)' else f'Sector: {agg_level}'}")

        prob_data = df_streaks.groupby('streak_days').agg(
            total_up=('up_count', 'sum'),
            total_down=('down_count', 'sum'),
            total_days=('total_days', 'sum'),
        ).reset_index()
        total_stock_days = max(df_streaks.groupby('Symbol')['total_days'].first().sum(), 1)
        prob_data['P(UP streak)'] = (prob_data['total_up'] / total_stock_days * 100).round(2)
        prob_data['P(DOWN streak)'] = (prob_data['total_down'] / total_stock_days * 100).round(2)
        prob_display = prob_data[['streak_days', 'total_up', 'P(UP streak)', 'total_down', 'P(DOWN streak)']].copy()
        prob_display.columns = ['Consecutive Days', 'UP Events', 'P(UP) %', 'DOWN Events', 'P(DOWN) %']
        prob_display_up = prob_display[['Consecutive Days', 'UP Events', 'P(UP) %']]
        prob_display_down = prob_display[['Consecutive Days', 'DOWN Events', 'P(DOWN) %']]
        prob_display_up, prob_display_down = with_row_ref(prob_display_up), with_row_ref(prob_display_down)
        prob_display_up, prob_display_down = format_dataframe(prob_display_up), format_dataframe(prob_display_down)
        prob_display_up, prob_display_down = add_header_index(prob_display_up), add_header_index(prob_display_down)
        # prob_display_styled = prob_display.style.pipe(center_top_index)

        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown("🟢 **UP Streak Probabilities**")
            st.dataframe(prob_display_up, use_container_width=True, hide_index=True)
        with pc2:
            st.markdown("🔴 **DOWN Streak Probabilities**")
            st.dataframe(prob_display_down, use_container_width=True, hide_index=True)

        if len(df_returns) > 0:
            ret_summary = df_returns.groupby(['streak_days', 'direction', 'return_window']).agg(
                avg_stock=('stock_return', 'mean'),
                avg_nasdaq=('nasdaq_return', 'mean'),
                count=('stock_return', 'count'),
            ).reset_index()
            ret_summary['excess'] = (ret_summary['avg_stock'] - ret_summary['avg_nasdaq']).round(1).astype(str) 
            ret_summary['avg_stock'] = ret_summary['avg_stock'].round(1)
            ret_summary['avg_nasdaq'] = ret_summary['avg_nasdaq'].round(1)
            ret_summary['Label'] = ret_summary.apply(
                lambda r: f"{'↑' if r['direction']=='up' else '↓'} {r['streak_days']}d → {r['return_window']}d", axis=1
            )
            ret_display = ret_summary[['Label', 'count', 'avg_stock', 'avg_nasdaq', 'excess']].copy()
            ret_display.columns = ['Event → Window', '# Events', 'Avg Stock Return %', 'Avg NASDAQ Return %', 'Excess Return %']
            st.dataframe(with_row_ref(ret_display), use_container_width=True, hide_index=True, height=360)

            st.markdown("**Component 3: Candlestick (Aggregate Average Returns)**")
            cand_src = df_returns[(df_returns['streak_days'] == streak_len) & (df_returns['return_window'] == return_days)].copy()
            if selected_direction in ['up', 'down']:
                cand_src = cand_src[cand_src['direction'] == selected_direction]
            if len(cand_src) > 0:
                stock_vals = cand_src['stock_return'].dropna()
                idx_vals = cand_src['nasdaq_return'].dropna()
                if len(stock_vals) > 0 and len(idx_vals) > 0:
                    fig_candle = go.Figure()
                    fig_candle.add_trace(go.Candlestick(x=['Aggregate Stock Return'], open=[0.0], high=[float(stock_vals.max())], low=[float(stock_vals.min())], close=[float(stock_vals.mean())], name='Stock'))
                    fig_candle.add_trace(go.Candlestick(x=['Aggregate NASDAQ Return'], open=[0.0], high=[float(idx_vals.max())], low=[float(idx_vals.min())], close=[float(idx_vals.mean())], name='NASDAQ'))
                    fig_candle.update_layout(template='plotly_dark', height=420, yaxis_title="Return (%)", margin=dict(l=60, r=40, t=60, b=40))
                    st.plotly_chart(fig_candle, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Annual Returns Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Annual Return Pattern Analysis</div>', unsafe_allow_html=True)
    st.caption("Stocks hitting +/-50% annual return thresholds over one-year or two-year windows.")

    df_yearly = compute_year_bound_prices(df_stock)
    df_idx_year = compute_index_year_returns(df_idx)

    # Base metadata (drop missing Company as required)
    df_meta = df_mc[['Symbol', 'Company', 'Market Cap $', 'Sector', 'Industry']].copy()
    df_meta = df_meta.dropna(subset=['Company'])
    df_meta = df_meta[df_meta['Company'].astype(str).str.strip() != ""]

    # Current and 52-week percentile
    latest_px = df_stock.sort_values('Date').groupby('Symbol').tail(1)[['Symbol', 'Close']].rename(
        columns={'Close': 'Current Price'}
    )
    p52_rows = []
    for sym, grp in df_stock.groupby('Symbol'):
        p52_rows.append({'Symbol': sym, '52 Week Percentile': percentile_52w(grp)})
    df_p52 = pd.DataFrame(p52_rows)
    stats_tail = latest_px.merge(df_p52, on='Symbol', how='left')

    # Build 1Y event tables (Table 1A / 1B)
    one_y = df_yearly.rename(columns={
        'Year': 'Year [1]',
        'Stock Return': 'Stock Return [Year 1]',
        'Start Price': 'Start Price',
        'End Price': 'End Price',
    }).copy()
    y2 = df_yearly[['Symbol', 'Year', 'Stock Return']].copy()
    y2['Year [1]'] = y2['Year'] - 1
    y2.rename(columns={'Stock Return': 'Stock Return [Year 2]'}, inplace=True)
    one_y = one_y.merge(y2[['Symbol', 'Year [1]', 'Stock Return [Year 2]']], on=['Symbol', 'Year [1]'], how='left')

    idx_y1 = df_idx_year.rename(columns={'Year': 'Year [1]', 'Nasdaq Return': 'Nasdaq Return [Year 1]'})
    idx_y2 = df_idx_year.copy()
    idx_y2['Year [1]'] = idx_y2['Year'] - 1
    idx_y2.rename(columns={'Nasdaq Return': 'Nasdaq Return [Year 2]'}, inplace=True)
    one_y = one_y.merge(idx_y1[['Year [1]', 'Nasdaq Return [Year 1]']], on='Year [1]', how='left')
    one_y = one_y.merge(idx_y2[['Year [1]', 'Nasdaq Return [Year 2]']], on='Year [1]', how='left')
    one_y = one_y.merge(df_meta, on='Symbol', how='inner').merge(stats_tail, on='Symbol', how='left')

    one_y_cols = [
        'Company', 'Symbol', 'Market Cap $', 'Sector', 'Industry', 'Year [1]',
        'Start Price', 'End Price', 'Stock Return [Year 1]', 'Nasdaq Return [Year 1]',
        'Stock Return [Year 2]', 'Nasdaq Return [Year 2]', 'Current Price', '52 Week Percentile'
    ]
    one_y = one_y[one_y_cols]

    t1a = one_y[one_y['Stock Return [Year 1]'] >= 50].sort_values(['Year [1]', 'Stock Return [Year 1]'], ascending=[False, False])
    t1b = one_y[one_y['Stock Return [Year 1]'] <= -50].sort_values(['Year [1]', 'Stock Return [Year 1]'], ascending=[False, True])

    # Build 2Y consecutive event tables (Table 2A / 2B)
    y1 = df_yearly[['Symbol', 'Year', 'Start Price', 'End Price', 'Stock Return']].copy()
    y2b = df_yearly[['Symbol', 'Year', 'Start Price', 'End Price', 'Stock Return']].copy()
    y2b['Year [1]'] = y2b['Year'] - 1
    y2b.rename(columns={
        'Year': 'Year [2]',
        'End Price': 'End Price [Year 2]',
        'Stock Return': 'Stock Return [Year 2]'
    }, inplace=True)
    two_y = y1.rename(columns={
        'Year': 'Year [1]',
        'Stock Return': 'Stock Return [Year 1]'
    }).merge(
        y2b[['Symbol', 'Year [1]', 'Year [2]', 'End Price [Year 2]', 'Stock Return [Year 2]']],
        on=['Symbol', 'Year [1]'],
        how='inner'
    )
    two_y['End Price'] = two_y['End Price [Year 2]']
    two_y['Combined Return [Year1+Year2]'] = ((1 + two_y['Stock Return [Year 1]'] / 100.0) * (1 + two_y['Stock Return [Year 2]'] / 100.0) - 1) * 100

    idx1 = df_idx_year.rename(columns={'Year': 'Year [1]', 'Nasdaq Return': 'Nasdaq Return [Year 1]'})
    idx2 = df_idx_year.copy()
    idx2['Year [1]'] = idx2['Year'] - 1
    idx2.rename(columns={'Nasdaq Return': 'Nasdaq Return [Year 2]'}, inplace=True)
    idx3 = df_idx_year.copy()
    idx3['Year [1]'] = idx3['Year'] - 2
    idx3.rename(columns={'Nasdaq Return': 'Nasdaq Return [Year 3]'}, inplace=True)

    y3_stock = df_yearly[['Symbol', 'Year', 'Stock Return']].copy()
    y3_stock['Year [1]'] = y3_stock['Year'] - 2
    y3_stock.rename(columns={'Stock Return': 'Stock Return [Year 3]'}, inplace=True)

    two_y = two_y.merge(idx1[['Year [1]', 'Nasdaq Return [Year 1]']], on='Year [1]', how='left')
    two_y = two_y.merge(idx2[['Year [1]', 'Nasdaq Return [Year 2]']], on='Year [1]', how='left')
    two_y = two_y.merge(idx3[['Year [1]', 'Nasdaq Return [Year 3]']], on='Year [1]', how='left')
    two_y = two_y.merge(y3_stock[['Symbol', 'Year [1]', 'Stock Return [Year 3]']], on=['Symbol', 'Year [1]'], how='left')
    two_y = two_y.merge(df_meta, on='Symbol', how='inner').merge(stats_tail, on='Symbol', how='left')

    two_y_cols = [
        'Company', 'Symbol', 'Market Cap $', 'Sector', 'Industry', 'Year [1]',
        'Start Price', 'End Price', 'Stock Return [Year 1]', 'Stock Return [Year 2]',
        'Nasdaq Return [Year 1]', 'Nasdaq Return [Year 2]', 'Stock Return [Year 3]',
        'Nasdaq Return [Year 3]', 'Current Price', '52 Week Percentile', 'Combined Return [Year1+Year2]'
    ]
    two_y = two_y[two_y_cols]
    t2a = two_y[two_y['Combined Return [Year1+Year2]'] >= 50].sort_values(['Year [1]', 'Combined Return [Year1+Year2]'], ascending=[False, False])
    t2b = two_y[two_y['Combined Return [Year1+Year2]'] <= -50].sort_values(['Year [1]', 'Combined Return [Year1+Year2]'], ascending=[False, True])
    t2a = t2a.drop(columns=['Combined Return [Year1+Year2]'])
    t2b = t2b.drop(columns=['Combined Return [Year1+Year2]'])

    # Formatting
    def _fmt_annual(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        if 'Market Cap $' in df_out.columns:
            df_out['Market Cap $'] = df_out['Market Cap $'].apply(lambda x: format_number(x, 1))
        if 'Current Price' in df_out.columns:
            df_out['Current Price'] = df_out['Current Price'].apply(lambda x: f"${x:,.2f}" if not pd.isna(x) else "N/A")
        for c in [col for col in df_out.columns if 'Return' in col or 'Percentile' in col]:
            if c == 'Current Price':
                continue
            df_out[c] = df_out[c].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "N/A")
        for c in [col for col in df_out.columns if 'Price' in col and c != 'Current Price']:
            df_out[c] = df_out[c].apply(lambda x: f"${x:,.2f}" if not pd.isna(x) else "N/A")
        return df_out

    # t1a, t1b, t2a, t2b = map(_fmt_annual, [t1a, t1b, t2a, t2b])
    t1a, t1b, t2a, t2b = map(with_row_ref, [t1a, t1b, t2a, t2b])
    t1a['Year [1]'] = t1a['Year [1]'].astype(str)
    t1b['Year [1]'] = t1b['Year [1]'].astype(str)
    t2a['Year [1]'] = t2a['Year [1]'].astype(str)
    t2b['Year [1]'] = t2b['Year [1]'].astype(str)
    t1a, t1b, t2a, t2b = map(format_dataframe, [t1a, t1b, t2a, t2b])
    t1a, t1b, t2a, t2b = map(add_header_index, [t1a, t1b, t2a, t2b])
    t1a_styled = t1a.style.pipe(center_top_index)
    t1b_styled = t1b.style.pipe(center_top_index)
    t2a_styled = t2a.style.pipe(center_top_index)
    t2b_styled = t2b.style.pipe(center_top_index)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Table 1.A — +50% or More in Year [1]**")
        st.dataframe(t1a_styled, use_container_width=True, hide_index=True, height=360)
    with c2:
        st.markdown("**Table 1.B — -50% or Less in Year [1]**")
        st.dataframe(t1b_styled, use_container_width=True, hide_index=True, height=360)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Table 2.A — +50% or More Over Year [1]+[2]**")
        st.dataframe(t2a_styled, use_container_width=True, hide_index=True, height=360)
    with c4:
        st.markdown("**Table 2.B — -50% or Less Over Year [1]+[2]**")
        st.dataframe(t2b_styled, use_container_width=True, hide_index=True, height=360)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption("NASDAQ Stock Streak Screener · Data as of 2026-02-11")
