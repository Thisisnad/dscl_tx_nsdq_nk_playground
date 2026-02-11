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
# Use current directory for cloud deployment (files should be in same directory as script)
DATA_DIR = Path(__file__).resolve().parent

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
    # 1. Market cap data
    mc_path = DATA_DIR / "NK_market_cap_20260211.xlsx"
    df_mc = pd.read_excel(mc_path, sheet_name="NASDAQ_screener")
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
    stock_path = DATA_DIR / "NK_stock_data_20260211.csv"
    df_stock = pd.read_csv(stock_path, parse_dates=['Date'])
    df_stock.columns = df_stock.columns.str.strip()
    df_stock = df_stock.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    
    # 3. NASDAQ Index data
    # Try multiple extensions
    idx_base = "NK_nsdq_index_20260211"
    idx_path = None
    for ext in ['.csv', '.xlsx', '']:
        p = DATA_DIR / (idx_base + ext)
        if p.exists():
            idx_path = p
            break
    
    if idx_path is None:
        # Try without extension (could be csv without extension)
        idx_path = DATA_DIR / idx_base
    
    if str(idx_path).endswith('.xlsx'):
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


# ─── Main App ────────────────────────────────────────────────────────────────

st.markdown("# 📈 NASDAQ Stock Streak Screener")
st.markdown("*Screen NASDAQ stocks for consecutive price streaks and analyze forward returns vs. the NASDAQ Composite Index*")

if not data_loaded:
    st.error(f"❌ **Failed to load data:** {load_error}")
    st.info("Ensure all data files are in the expected directory and file names match.")
    st.stop()

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Stock Price History",
    "🔍 Stock Returns Streak Analysis",
    "📉 Macro Level Analysis"
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
        c6.metric("Date Range", f"{first_date.strftime('%Y-%m-%d')} → {latest_date.strftime('%Y-%m-%d')}")
        c7.metric("Trading Days", f"{len(df_sel):,}")
        c8.metric("Min Price", f"${min_price:,.2f} ({min_date.strftime('%Y-%m-%d')})")
        c9.metric("Max Price", f"${max_price:,.2f} ({max_date.strftime('%Y-%m-%d')})")
        
        # Price History Table
        st.markdown('<div class="section-header">Closing Price History</div>', unsafe_allow_html=True)
        df_display = df_sel[['Date', 'Close']].copy()
        df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d')
        df_display = df_display.sort_values('Date', ascending=False)
        st.dataframe(df_display, use_container_width=True, height=300)
        
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
    m2.metric("Date Range Start", date_min.strftime('%Y-%m-%d'))
    m3.metric("Date Range End", date_max.strftime('%Y-%m-%d'))
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
        st.dataframe(mc_dist, use_container_width=True, hide_index=True)
    
    with pop_col2:
        # Sector distribution
        st.markdown("**Sector Distribution**")
        sector_dist = df_mc['Sector'].value_counts().reset_index()
        sector_dist.columns = ['Sector', 'Count']
        st.dataframe(sector_dist, use_container_width=True, hide_index=True)
    
    # Industry distribution (collapsible)
    with st.expander("Industry Distribution", expanded=False):
        ind_dist = df_mc['Industry'].value_counts().reset_index()
        ind_dist.columns = ['Industry', 'Count']
        st.dataframe(ind_dist, use_container_width=True, hide_index=True, height=400)
    
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
            ['5 days', '7 days', '10 days'],
            index=0, key='event_type'
        )
        streak_len = int(event_type.split()[0])
    with fc6:
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
        
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown(f"**🟢 {streak_len}-Day UP Streaks (Last 14 Days)**")
            df_recent_up = build_recent_table(recent_up, 'UP')
            if len(df_recent_up) > 0:
                st.dataframe(df_recent_up, use_container_width=True, hide_index=True)
            else:
                st.info("No UP streak events in the last 14 days.")
        
        with r_col2:
            st.markdown(f"**🔴 {streak_len}-Day DOWN Streaks (Last 14 Days)**")
            df_recent_down = build_recent_table(recent_down, 'DOWN')
            if len(df_recent_down) > 0:
                st.dataframe(df_recent_down, use_container_width=True, hide_index=True)
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
            summary['earliest'] = summary['earliest'].dt.strftime('%Y-%m-%d')
            summary['latest'] = summary['latest'].dt.strftime('%Y-%m-%d')
            
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
        st.markdown(f"**🟢 {streak_len}-Day Consecutive UP Streaks → {return_days}-Day Forward Returns**")
        df_summary_up = build_summary(all_events_up, 'UP')
        if len(df_summary_up) > 0:
            st.metric("Total UP Events", f"{len(all_events_up):,} across {df_summary_up['Symbol'].nunique()} stocks")
            st.dataframe(df_summary_up, use_container_width=True, hide_index=True, height=400)
        else:
            st.info("No UP streak events found with current filters.")
        
        # DOWN streaks summary
        st.markdown(f"**🔴 {streak_len}-Day Consecutive DOWN Streaks → {return_days}-Day Forward Returns**")
        df_summary_down = build_summary(all_events_down, 'DOWN')
        if len(df_summary_down) > 0:
            st.metric("Total DOWN Events", f"{len(all_events_down):,} across {df_summary_down['Symbol'].nunique()} stocks")
            st.dataframe(df_summary_down, use_container_width=True, hide_index=True, height=400)
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
        st.stop()
    
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
    
    with st.spinner("Computing macro statistics..."):
        time_start_str2 = str(time_start) if time_start else None
        stock_hash2 = f"{len(df_stock)}_{df_stock['Date'].max()}"
        all_streak_data, all_return_data = compute_macro_stats(
            tuple(in_scope_symbols), time_start_str2, stock_hash2
        )
    
    if not all_streak_data:
        st.warning("Insufficient data for macro analysis.")
        st.stop()
    
    df_streaks_all = pd.DataFrame(all_streak_data)
    df_returns_all = pd.DataFrame(all_return_data) if all_return_data else pd.DataFrame()
    
    def render_macro_section(df_streaks, df_returns, section_label):
        """Render probability tables, return tables, and charts for a subset."""
        
        st.markdown(f"#### {section_label}")
        
        # 1. Probability Tables
        st.markdown("**Probability of Consecutive Price Movement (0-10 Days)**")
        
        # Aggregate by streak_days
        prob_data = df_streaks.groupby('streak_days').agg(
            total_up=('up_count', 'sum'),
            total_down=('down_count', 'sum'),
            total_days=('total_days', 'sum'),
        ).reset_index()
        
        # Probability = events / (total trading days across all stocks)
        total_stock_days = df_streaks.groupby('Symbol')['total_days'].first().sum()
        prob_data['P(UP streak)'] = (prob_data['total_up'] / total_stock_days * 100).round(2)
        prob_data['P(DOWN streak)'] = (prob_data['total_down'] / total_stock_days * 100).round(2)
        prob_data['UP Events'] = prob_data['total_up']
        prob_data['DOWN Events'] = prob_data['total_down']
        
        prob_display = prob_data[['streak_days', 'UP Events', 'P(UP streak)', 'DOWN Events', 'P(DOWN streak)']].copy()
        prob_display.columns = ['Consecutive Days', 'UP Events', 'P(UP) %', 'DOWN Events', 'P(DOWN) %']
        
        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown("🟢 **UP Streak Probabilities**")
            st.dataframe(prob_display[['Consecutive Days', 'UP Events', 'P(UP) %']], 
                        use_container_width=True, hide_index=True)
        with pc2:
            st.markdown("🔴 **DOWN Streak Probabilities**")
            st.dataframe(prob_display[['Consecutive Days', 'DOWN Events', 'P(DOWN) %']], 
                        use_container_width=True, hide_index=True)
        
        # Probability chart
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Bar(
            x=prob_display['Consecutive Days'], y=prob_display['P(UP) %'],
            name='UP Streak', marker_color='#00c853'
        ))
        fig_prob.add_trace(go.Bar(
            x=prob_display['Consecutive Days'], y=prob_display['P(DOWN) %'],
            name='DOWN Streak', marker_color='#ff1744'
        ))
        fig_prob.update_layout(
            template='plotly_dark', height=350, barmode='group',
            title="Streak Probability by Consecutive Days",
            xaxis_title="Consecutive Days", yaxis_title="Probability (%)",
            margin=dict(l=60, r=40, t=60, b=40),
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # 2. Return Comparison Table
        if len(df_returns) > 0:
            st.markdown("**Average Returns: Stock vs NASDAQ Composite**")
            
            ret_summary = df_returns.groupby(['streak_days', 'direction', 'return_window']).agg(
                avg_stock=('stock_return', 'mean'),
                avg_nasdaq=('nasdaq_return', 'mean'),
                count=('stock_return', 'count'),
            ).reset_index()
            
            ret_summary['avg_stock'] = ret_summary['avg_stock'].round(1)
            ret_summary['avg_nasdaq'] = ret_summary['avg_nasdaq'].round(1)
            ret_summary['excess'] = (ret_summary['avg_stock'] - ret_summary['avg_nasdaq']).round(1)
            
            ret_pivot = ret_summary.copy()
            ret_pivot['Label'] = ret_pivot.apply(
                lambda r: f"{'↑' if r['direction']=='up' else '↓'} {r['streak_days']}d → {r['return_window']}d", axis=1
            )
            
            ret_display = ret_pivot[['Label', 'count', 'avg_stock', 'avg_nasdaq', 'excess']].copy()
            ret_display.columns = ['Event → Window', '# Events', 'Avg Stock Return %', 'Avg NASDAQ Return %', 'Excess Return %']
            st.dataframe(ret_display, use_container_width=True, hide_index=True, height=400)
            
            # 3. Visualisations
            st.markdown("**Visualisations**")
            
            # Heatmap: Average excess return by streak length and return window
            for direction in ['up', 'down']:
                dir_label = "UP" if direction == 'up' else "DOWN"
                dir_data = ret_summary[ret_summary['direction'] == direction]
                if len(dir_data) == 0:
                    continue
                
                heatmap_data = dir_data.pivot_table(
                    index='streak_days', columns='return_window', values='excess', aggfunc='mean'
                )
                
                if len(heatmap_data) > 0:
                    fig_heat = px.imshow(
                        heatmap_data.values,
                        x=[f"{c}d" for c in heatmap_data.columns],
                        y=[f"{r}d streak" for r in heatmap_data.index],
                        color_continuous_scale='RdYlGn',
                        labels=dict(x="Return Window", y="Streak Length", color="Excess Return %"),
                        title=f"{'🟢' if direction == 'up' else '🔴'} {dir_label} Streaks — Excess Return vs NASDAQ (%)",
                        text_auto='.1f',
                    )
                    fig_heat.update_layout(
                        template='plotly_dark', height=400,
                        margin=dict(l=80, r=40, t=60, b=40),
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
            
            # Bar chart: Stock vs NASDAQ returns for selected streak
            for direction in ['up', 'down']:
                dir_label = "UP" if direction == 'up' else "DOWN"
                dir_data = ret_summary[(ret_summary['direction'] == direction) & (ret_summary['streak_days'] == streak_len)]
                if len(dir_data) == 0:
                    continue
                
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=[f"{r}d" for r in dir_data['return_window']],
                    y=dir_data['avg_stock'],
                    name='Avg Stock Return', marker_color='#00d4ff',
                ))
                fig_bar.add_trace(go.Bar(
                    x=[f"{r}d" for r in dir_data['return_window']],
                    y=dir_data['avg_nasdaq'],
                    name='Avg NASDAQ Return', marker_color='#ff6b6b',
                ))
                fig_bar.update_layout(
                    template='plotly_dark', height=350, barmode='group',
                    title=f"{'🟢' if direction == 'up' else '🔴'} After {streak_len}-Day {dir_label} Streak: Avg Returns",
                    xaxis_title="Return Window", yaxis_title="Avg Return (%)",
                    margin=dict(l=60, r=40, t=60, b=40),
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Distribution plot: stock returns after streak events
            for direction in ['up', 'down']:
                dir_label = "UP" if direction == 'up' else "DOWN"
                dir_returns = df_returns[
                    (df_returns['direction'] == direction) &
                    (df_returns['streak_days'] == streak_len) &
                    (df_returns['return_window'] == return_days)
                ]
                if len(dir_returns) > 5:
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=dir_returns['stock_return'], name='Stock Return',
                        marker_color='#00d4ff', opacity=0.7, nbinsx=40,
                    ))
                    fig_dist.add_trace(go.Histogram(
                        x=dir_returns['nasdaq_return'], name='NASDAQ Return',
                        marker_color='#ff6b6b', opacity=0.7, nbinsx=40,
                    ))
                    fig_dist.update_layout(
                        template='plotly_dark', height=350, barmode='overlay',
                        title=f"Return Distribution: {streak_len}d {dir_label} Streak → {return_days}d Returns",
                        xaxis_title="Return (%)", yaxis_title="Frequency",
                        margin=dict(l=60, r=40, t=60, b=40),
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            # Scatter: Stock return vs NASDAQ return
            scatter_data = df_returns[
                (df_returns['streak_days'] == streak_len) &
                (df_returns['return_window'] == return_days)
            ]
            if len(scatter_data) > 5:
                fig_scatter = px.scatter(
                    scatter_data, x='nasdaq_return', y='stock_return',
                    color='direction',
                    color_discrete_map={'up': '#00c853', 'down': '#ff1744'},
                    title=f"Stock vs NASDAQ {return_days}d Return (After {streak_len}d Streak)",
                    labels={'nasdaq_return': 'NASDAQ Return (%)', 'stock_return': 'Stock Return (%)'},
                    opacity=0.5,
                )
                fig_scatter.add_shape(type='line', x0=-50, y0=-50, x1=100, y1=100,
                                     line=dict(color='gray', dash='dash'))
                fig_scatter.update_layout(
                    template='plotly_dark', height=450,
                    margin=dict(l=60, r=40, t=60, b=40),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Render based on aggregation level
    if agg_level == 'All Stocks (Aggregate)':
        render_macro_section(df_streaks_all, df_returns_all, "All Stocks in Scope")
    else:
        # Filter to selected sector
        df_s_sec = df_streaks_all[df_streaks_all['Sector'] == agg_level]
        df_r_sec = df_returns_all[df_returns_all['Sector'] == agg_level] if len(df_returns_all) > 0 else pd.DataFrame()
        render_macro_section(df_s_sec, df_r_sec, f"Sector: {agg_level}")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption("NASDAQ Stock Streak Screener · Data as of 2026-02-11")
