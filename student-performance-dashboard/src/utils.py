import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def apply_custom_css():
    st.markdown("""
    <style>
        /* Base typography & aesthetics */
        .main {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
            margin-bottom: 12px;
            transition: all 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: #475569;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        }
        .metric-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94a3b8;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 2.0rem;
            font-weight: 800;
            color: #f8fafc;
            line-height: 1.1;
        }
        .metric-subtitle {
            font-size: 0.8rem;
            margin-top: 6px;
            font-weight: 500;
        }

        /* Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 40%, #312e81 100%);
            color: white;
            padding: 24px 28px;
            border-radius: 16px;
            margin-bottom: 20px;
            box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.35);
        }
        .hero-title {
            font-size: 2.0rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 6px;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            opacity: 0.92;
            line-height: 1.5;
        }

        /* Insight Callout Box */
        .insight-box {
            background: #0f172a;
            border-left: 4px solid #38bdf8;
            border-radius: 8px;
            padding: 12px 16px;
            margin: 12px 0 18px 0;
            font-size: 0.9rem;
            color: #cbd5e1;
            line-height: 1.5;
        }
        .insight-title {
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 4px;
        }

        /* Grade Legend Card */
        .legend-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 12px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
        }
        .legend-item {
            display: inline-flex;
            align-items: center;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .legend-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }

        /* Student Report Card (Tab 3) */
        .report-card {
            background: linear-gradient(145deg, #1e293b, #0f172a);
            border: 2px solid #3b82f6;
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        }
        .report-grade-badge {
            font-size: 2.6rem;
            font-weight: 800;
            margin: 10px 0;
            letter-spacing: -0.02em;
        }
    </style>
    """, unsafe_allow_html=True)

def render_kpi(col, title, value, subtitle=None, color="#38bdf8"):
    sub_html = f'<div class="metric-subtitle" style="color: {color};">{subtitle}</div>' if subtitle else ''
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)

def render_hero_banner():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🎓 Student Performance Prediction & Analytics Dashboard</div>
        <div class="hero-subtitle">
            An intuitive educational intelligence platform: discover behavioral drivers of academic success, 
            forecast final grades with machine learning, and design timely interventions for at-risk students.
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_insight_box(title, message):
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title">💡 Key Takeaway: {title}</div>
        <div>{message}</div>
    </div>
    """, unsafe_allow_html=True)

def render_grading_legend():
    st.markdown("""
    <div class="legend-card">
        <span style="font-size: 0.85rem; font-weight: 700; color: #94a3b8;">📖 Grading Scale Reference (Portugal Standard 0-20):</span>
        <span class="legend-item" style="color: #10b981;"><span class="legend-dot" style="background:#10b981;"></span>16 - 20: Grade A (Excellent)</span>
        <span class="legend-item" style="color: #3b82f6;"><span class="legend-dot" style="background:#3b82f6;"></span>14 - 15: Grade B (Good)</span>
        <span class="legend-item" style="color: #6366f1;"><span class="legend-dot" style="background:#6366f1;"></span>12 - 13: Grade C (Satisfactory)</span>
        <span class="legend-item" style="color: #f59e0b;"><span class="legend-dot" style="background:#f59e0b;"></span>10 - 11: Grade D (Sufficient / Pass)</span>
        <span class="legend-item" style="color: #ef4444;"><span class="legend-dot" style="background:#ef4444;"></span>&lt; 10: Grade F (Failing / At-Risk)</span>
    </div>
    """, unsafe_allow_html=True)

def get_plotly_theme():
    return {
        'template': 'plotly_dark',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(15,23,42,0.6)',
        'font': {'family': '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto', 'color': '#cbd5e1'}
    }
