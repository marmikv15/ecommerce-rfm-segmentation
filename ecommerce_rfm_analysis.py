"""
=============================================================
PROJECT 2: E-Commerce Customer Segmentation — RFM Analysis
=============================================================
Author: Marmik Vyas
Tools: Python · Pandas · Matplotlib · Seaborn · Scikit-Learn
Dataset: UK Online Retail Dataset (simulated, based on real patterns)

Business Problem:
    An e-commerce company wants to understand which customers
    are most valuable, which are at risk of churning, and what
    marketing actions to take for each customer segment.

BA Framework Applied:
    → Business Problem Definition
    → Requirements Elicitation (implicit: what does marketing need?)
    → RFM Model Specification
    → Segment-to-Action Mapping (BA Deliverable)
    → KPI Dashboard for Marketing Leadership

RFM Methodology:
    R = Recency     (days since last purchase — lower is better)
    F = Frequency   (number of purchases — higher is better)
    M = Monetary    (total spend — higher is better)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'primary': '#2E75B6', 'secondary': '#ED7D31', 'success': '#70AD47',
          'danger': '#C00000', 'warning': '#FFC000', 'purple': '#7030A0', 'teal': '#00B0A0'}

print("=" * 65)
print("E-COMMERCE CUSTOMER SEGMENTATION — RFM ANALYSIS")
print("Business Analyst: Marmik Vyas")
print("=" * 65)

# ─── DATA GENERATION ─────────────────────────────────────
np.random.seed(99)
n_customers = 2000


customer_ids = [f'CUST-{str(i).zfill(5)}' for i in range(1, n_customers + 1)]
countries = np.random.choice(['UK', 'Germany', 'France', 'Spain', 'Netherlands', 'Belgium'],
                              n_customers, p=[0.60, 0.15, 0.10, 0.07, 0.05, 0.03])


transactions = []
reference_date = datetime(2024, 12, 31)

for i, (cid, country) in enumerate(zip(customer_ids, countries)):
    
    ctype = np.random.choice(['champion', 'loyal', 'at_risk', 'lost', 'new'],
                              p=[0.15, 0.25, 0.25, 0.20, 0.15])

    if ctype == 'champion':
        n_orders = np.random.randint(15, 50)
        last_days_ago = np.random.randint(1, 30)
        avg_spend = np.random.uniform(200, 800)
    elif ctype == 'loyal':
        n_orders = np.random.randint(6, 15)
        last_days_ago = np.random.randint(10, 60)
        avg_spend = np.random.uniform(100, 300)
    elif ctype == 'at_risk':
        n_orders = np.random.randint(3, 8)
        last_days_ago = np.random.randint(60, 180)
        avg_spend = np.random.uniform(80, 250)
    elif ctype == 'lost':
        n_orders = np.random.randint(1, 4)
        last_days_ago = np.random.randint(180, 365)
        avg_spend = np.random.uniform(30, 100)
    else:  # new
        n_orders = np.random.randint(1, 3)
        last_days_ago = np.random.randint(1, 45)
        avg_spend = np.random.uniform(50, 200)

    for _ in range(n_orders):
        order_date = reference_date - timedelta(days=last_days_ago + np.random.randint(0, 300))
        spend = max(5, np.random.normal(avg_spend, avg_spend * 0.3))
        transactions.append({'CustomerID': cid, 'Country': country,
                              'OrderDate': order_date, 'OrderValue': round(spend, 2),
                              'CustomerType_True': ctype})

df_trans = pd.DataFrame(transactions)
df_trans['OrderDate'] = pd.to_datetime(df_trans['OrderDate'])

print(f"\n✅ Transaction Data Generated:")
print(f"   → Total Transactions: {len(df_trans):,}")
print(f"   → Unique Customers:   {df_trans['CustomerID'].nunique():,}")
print(f"   → Date Range:         {df_trans['OrderDate'].min().date()} to {df_trans['OrderDate'].max().date()}")

# ───RFM CALCULATION ──────────────────────────────────────
print("\n" + "─" * 65)
print("SECTION 1: RFM SCORE CALCULATION")
print("─" * 65)

rfm = df_trans.groupby('CustomerID').agg(
    Recency=('OrderDate', lambda x: (reference_date - x.max()).days),
    Frequency=('OrderDate', 'count'),
    Monetary=('OrderValue', 'sum'),
    Country=('Country', 'first')
).reset_index()

rfm['Monetary'] = rfm['Monetary'].round(2)


rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm['RFM_Score'] = rfm['R_Score'] * 100 + rfm['F_Score'] * 10 + rfm['M_Score']
rfm['RFM_Total'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

# ───CUSTOMER SEGMENTATION ────────────────────────────────
def assign_segment(row):
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    total = row['RFM_Total']
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and total >= 11:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'New Customers'
    elif r <= 2 and f >= 3 and m >= 3:
        return 'At Risk'
    elif r <= 2 and f >= 4 and m >= 4:
        return 'Cannot Lose Them'
    elif r <= 2 and f <= 2:
        return 'Lost'
    elif r >= 3 and f >= 2 and total >= 8:
        return 'Potential Loyalists'
    else:
        return 'Needs Attention'

rfm['Segment'] = rfm.apply(assign_segment, axis=1)

segment_summary = rfm.groupby('Segment').agg(
    Customer_Count=('CustomerID', 'count'),
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean'),
    Total_Revenue=('Monetary', 'sum')
).round(2)
segment_summary['Pct_Customers'] = (segment_summary['Customer_Count'] / len(rfm) * 100).round(1)
segment_summary['Pct_Revenue'] = (segment_summary['Total_Revenue'] / rfm['Monetary'].sum() * 100).round(1)
segment_summary = segment_summary.sort_values('Total_Revenue', ascending=False)

print("\nCustomer Segment Summary:")
print(segment_summary[['Customer_Count', 'Pct_Customers', 'Avg_Recency', 'Avg_Frequency',
                         'Avg_Monetary', 'Total_Revenue', 'Pct_Revenue']].to_string())

# ───RETENTION STRATEGY MAPPING (BA DELIVERABLE) ─────────
print("\n" + "─" * 65)
print("SECTION 2: SEGMENT-TO-ACTION MAPPING (BA Deliverable)")
print("─" * 65)

strategies = {
    'Champions':           {'Priority': '🔴 High',  'Action': 'Reward + Upsell premium products',
                            'Channel': 'Email + App Push', 'Goal': 'Maintain spend frequency'},
    'Loyal Customers':     {'Priority': '🔴 High',  'Action': 'Loyalty points + Early access offers',
                            'Channel': 'Email + SMS', 'Goal': 'Increase basket size (M↑)'},
    'At Risk':             {'Priority': '🔴 High',  'Action': 'Win-back campaign with 15% discount',
                            'Channel': 'Email + Retargeting', 'Goal': 'Re-engage before lost'},
    'Cannot Lose Them':    {'Priority': '🔴 Critical', 'Action': 'Personal outreach + premium offer',
                            'Channel': 'Phone + Email', 'Goal': 'Prevent churn immediately'},
    'Potential Loyalists': {'Priority': '🟡 Medium', 'Action': 'Membership/subscription offer',
                            'Channel': 'Email + In-app', 'Goal': 'Increase frequency (F↑)'},
    'New Customers':       {'Priority': '🟡 Medium', 'Action': 'Onboarding series + product education',
                            'Channel': 'Email drip', 'Goal': 'Drive second purchase'},
    'Needs Attention':     {'Priority': '🟢 Low',   'Action': 'Targeted campaign based on past category',
                            'Channel': 'Email', 'Goal': 'Re-engage'},
    'Lost':                {'Priority': '🟢 Low',   'Action': 'Reactivation email with strong offer',
                            'Channel': 'Email only', 'Goal': 'Last-chance re-activation'},
}

strat_df = pd.DataFrame(strategies).T
print(strat_df.to_string())

# ───VISUALIZATIONS ───────────────────────────────────────
fig = plt.figure(figsize=(20, 15))
fig.suptitle('E-Commerce Customer Segmentation — RFM Analysis Dashboard\nBusiness Analyst: Marmik Vyas',
             fontsize=16, fontweight='bold', y=0.98)

seg_colors = {'Champions': '#C00000', 'Loyal Customers': '#ED7D31', 'At Risk': '#FFC000',
              'Cannot Lose Them': '#7030A0', 'Potential Loyalists': '#2E75B6',
              'New Customers': '#70AD47', 'Needs Attention': '#00B0A0', 'Lost': '#A5A5A5'}

# Plot 1: Segment Distribution (Treemap-style bar)
ax1 = fig.add_subplot(3, 3, 1)
seg_rev = segment_summary['Total_Revenue'].sort_values(ascending=False)
bar_colors = [seg_colors.get(s, '#2E75B6') for s in seg_rev.index]
bars = ax1.barh(seg_rev.index, seg_rev.values / 1e3, color=bar_colors)
ax1.set_title('Revenue by Segment (£K)', fontweight='bold', fontsize=11)
for bar in bars:
    ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f'£{bar.get_width():.0f}K', va='center', fontsize=8)

# Plot 2: Customer Count by Segment
ax2 = fig.add_subplot(3, 3, 2)
seg_count = segment_summary['Customer_Count'].sort_values(ascending=False)
bar_colors2 = [seg_colors.get(s, '#2E75B6') for s in seg_count.index]
ax2.bar(range(len(seg_count)), seg_count.values, color=bar_colors2)
ax2.set_xticks(range(len(seg_count)))
ax2.set_xticklabels(seg_count.index, rotation=35, ha='right', fontsize=8)
ax2.set_title('Customer Count by Segment', fontweight='bold', fontsize=11)

# Plot 3: Revenue % Pie
ax3 = fig.add_subplot(3, 3, 3)
top5 = segment_summary.nlargest(5, 'Total_Revenue')
others_rev = segment_summary['Total_Revenue'].sum() - top5['Total_Revenue'].sum()
pie_labels = list(top5.index) + ['Others']
pie_values = list(top5['Total_Revenue']) + [others_rev]
pie_colors = [seg_colors.get(s, '#A5A5A5') for s in top5.index] + ['#D3D3D3']
ax3.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', colors=pie_colors,
        startangle=90, textprops={'fontsize': 8})
ax3.set_title('Revenue Share by Segment', fontweight='bold', fontsize=11)

# Plot 4: RFM Scatter — Recency vs Monetary
ax4 = fig.add_subplot(3, 3, 4)
for seg, grp in rfm.groupby('Segment'):
    ax4.scatter(grp['Recency'], grp['Monetary'], c=seg_colors.get(seg, '#999'),
                label=seg, alpha=0.5, s=15)
ax4.set_xlabel('Recency (days)')
ax4.set_ylabel('Monetary (£)')
ax4.set_title('Recency vs Monetary by Segment', fontweight='bold', fontsize=11)
ax4.legend(fontsize=6, loc='upper right')

# Plot 5: Avg Frequency by Segment
ax5 = fig.add_subplot(3, 3, 5)
avg_freq = segment_summary['Avg_Frequency'].sort_values(ascending=False)
bar_colors5 = [seg_colors.get(s, '#2E75B6') for s in avg_freq.index]
ax5.bar(range(len(avg_freq)), avg_freq.values, color=bar_colors5)
ax5.set_xticks(range(len(avg_freq)))
ax5.set_xticklabels(avg_freq.index, rotation=35, ha='right', fontsize=8)
ax5.set_title('Avg Order Frequency by Segment', fontweight='bold', fontsize=11)

# Plot 6: R/F/M Score Heatmap
ax6 = fig.add_subplot(3, 3, 6)
rfm_means = rfm.groupby('Segment')[['R_Score', 'F_Score', 'M_Score']].mean().round(2)
sns.heatmap(rfm_means, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax6,
            linewidths=0.5, vmin=1, vmax=5, cbar_kws={'shrink': 0.8})
ax6.set_title('RFM Score Heatmap by Segment', fontweight='bold', fontsize=11)
ax6.tick_params(axis='y', labelsize=8)

# Plot 7: Revenue per Customer by Segment
ax7 = fig.add_subplot(3, 3, 7)
rev_per_cust = (segment_summary['Total_Revenue'] / segment_summary['Customer_Count']).sort_values(ascending=False)
bar_colors7 = [seg_colors.get(s, '#2E75B6') for s in rev_per_cust.index]
bars7 = ax7.barh(rev_per_cust.index, rev_per_cust.values, color=bar_colors7)
ax7.set_title('Avg Revenue per Customer (£)', fontweight='bold', fontsize=11)
for bar in bars7:
    ax7.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
             f'£{bar.get_width():.0f}', va='center', fontsize=8)

# Plot 8: Country Distribution
ax8 = fig.add_subplot(3, 3, 8)
country_rev = df_trans.groupby('Country')['OrderValue'].sum().sort_values(ascending=False)
bars8 = ax8.bar(country_rev.index, country_rev.values / 1e3, color=COLORS['primary'])
ax8.set_title('Revenue by Country (£K)', fontweight='bold', fontsize=11)
for bar in bars8:
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'£{bar.get_height():.0f}K', ha='center', fontsize=8)

# Plot 9: KPI Summary Box
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis('off')
total_rev = rfm['Monetary'].sum()
champions = rfm[rfm['Segment'] == 'Champions']
at_risk = rfm[rfm['Segment'] == 'At Risk']
kpi_text = f"""
KEY BUSINESS KPIs
─────────────────────────────
Total Customers:     {len(rfm):,}
Total Revenue:       £{total_rev:,.0f}

Champions:          {len(champions)} customers
→ {len(champions)/len(rfm)*100:.1f}% of base, {champions['Monetary'].sum()/total_rev*100:.1f}% of revenue

At Risk:            {len(at_risk)} customers
→ Revenue at stake: £{at_risk['Monetary'].sum():,.0f}

Avg Order Value:    £{rfm['Monetary'].mean():.2f}
Avg Frequency:      {rfm['Frequency'].mean():.1f} orders
Avg Recency:        {rfm['Recency'].mean():.0f} days
─────────────────────────────
Recommendation:
Re-engage 'At Risk' segment
to recover £{at_risk['Monetary'].sum():,.0f} revenue
"""
ax9.text(0.05, 0.95, kpi_text, transform=ax9.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#E8F4FD', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/claude/projects/2_ecommerce_rfm/rfm_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Dashboard saved: rfm_dashboard.png")

# Export
rfm.to_csv('/home/claude/projects/2_ecommerce_rfm/rfm_scores.csv', index=False)
df_trans.to_csv('/home/claude/projects/2_ecommerce_rfm/transactions.csv', index=False)
strat_df.to_csv('/home/claude/projects/2_ecommerce_rfm/retention_strategy_map.csv')
print("✅ CSVs exported: rfm_scores.csv, transactions.csv, retention_strategy_map.csv")
print("\n" + "=" * 65)
print("ANALYSIS COMPLETE")
print("=" * 65)
