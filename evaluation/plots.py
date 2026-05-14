import matplotlib.pyplot as plt
import numpy as np

# --- 1. ΔΕΔΟΜΕΝΑ ΑΠΟ ΤΑ RUNS ΣΟΥ ---
profiles = [
    "Normal", "Commuter", "Saturday", "Sunday", "Low", 
    "High Stress", "Flattened", "Bimodal", "Event", "Early Spike"
]

# Profit (€)
baseline_profit = [176455.12, 164030.23, 203721.40, 148616.41, 150168.56, 198033.68, 203298.05, 153844.82, 197492.97, 175389.56]
ai_profit       = [182029.67, 174788.57, 212414.85, 148467.36, 150690.50, 203309.77, 209687.46, 160691.33, 200121.42, 182638.42]

# Service Rate (%)
baseline_service = [77.7, 76.5, 68.6, 75.3, 100.0, 57.5, 88.6, 63.2, 82.1, 70.3]
ai_service       = [79.5, 80.1, 71.2, 75.7, 100.0, 58.8, 91.2, 65.5, 82.5, 73.2]

# Wait Time (minutes)
baseline_wait = [22.9, 16.9, 21.6, 19.4, 9.4, 23.7, 20.3, 17.1, 23.1, 24.2]
ai_wait       = [23.8, 17.4, 17.2, 18.7, 7.6, 23.6, 20.3, 17.1, 23.3, 24.0]

# --- 2. ΡΥΘΜΙΣΕΙΣ ΓΡΑΦΗΜΑΤΩΝ ---
x = np.arange(len(profiles))
width = 0.35

# Χρώματα 
color_baseline = '#e74c3c'   # Κόκκινο Baseline
color_ai_profit = '#3498db'  # Μπλε AI (Κέρδος/Service)
color_ai_wait = '#2ecc71'    # Πράσινο AI (Χρόνος Αναμονής)

def add_full_labels(ax, rects_b, rects_a, data_b, data_a, format_func, is_lower_better=False):
    """Τυπώνει την ακριβή τιμή στο Baseline και Τιμή + % Διαφορά στο AI"""
    
    # Labels για το Baseline (Μόνο η ακριβής τιμή, μαύρο χρώμα)
    for i, rect in enumerate(rects_b):
        height = rect.get_height()
        val_str = format_func(data_b[i])
        ax.annotate(val_str,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='#333333')

    # Labels για το AI (Ακριβής τιμή + Ποσοστό, χρωματισμένα ανάλογα με την επιτυχία)
    for i, rect in enumerate(rects_a):
        height = rect.get_height()
        val_str = format_func(data_a[i])
        
        diff = ((data_a[i] - data_b[i]) / data_b[i]) * 100
        
        if is_lower_better:
            color = '#27ae60' if diff <= 0 else '#c0392b' # Πράσινο αν μειώθηκε ο χρόνος
            sign = "" if diff <= 0 else "+"
        else:
            color = '#27ae60' if diff >= 0 else '#c0392b' # Πράσινο αν αυξήθηκε το κέρδος
            sign = "+" if diff >= 0 else ""
            
        text = f"{val_str}\n({sign}{diff:.1f}%)"
        
        ax.annotate(text,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)

# Συναρτήσεις μορφοποίησης αριθμών
format_eur = lambda x: f"{int(x):,}€"
format_pct = lambda x: f"{x:.1f}%"
format_min = lambda x: f"{x:.1f}m"

# =====================================================================
# --- ΓΡΑΦΗΜΑ 1: ΚΕΡΔΟΣ ---
# =====================================================================
fig1, ax1 = plt.subplots(figsize=(14, 7))
rects1_b = ax1.bar(x - width/2, baseline_profit, width, label='Greedy Baseline', color=color_baseline, edgecolor='black', alpha=0.8)
rects1_a = ax1.bar(x + width/2, ai_profit, width, label='PPO AI Agent', color=color_ai_profit, edgecolor='black', alpha=0.8)

ax1.set_title('Σύγκριση Καθαρού Κέρδους (Net Profit) ανά Προφίλ', fontsize=14, fontweight='bold')
ax1.set_ylabel('Κέρδος (€)', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(profiles, fontsize=11, rotation=15)
# Αυξάνουμε το όριο του άξονα Y για να χωρέσουν τα διπλά ταμπελάκια
ax1.set_ylim(0, max(max(baseline_profit), max(ai_profit)) * 1.15) 
ax1.legend()
add_full_labels(ax1, rects1_b, rects1_a, baseline_profit, ai_profit, format_eur)

fig1.tight_layout()
fig1.savefig('thesis_profit_comparison.png', dpi=300, bbox_inches='tight')
print("Αποθηκεύτηκε: thesis_profit_comparison.png")

# =====================================================================
# --- ΓΡΑΦΗΜΑ 2: ΠΟΣΟΣΤΟ ΕΞΥΠΗΡΕΤΗΣΗΣ ---
# =====================================================================
fig2, ax2 = plt.subplots(figsize=(14, 7))
rects2_b = ax2.bar(x - width/2, baseline_service, width, label='Greedy Baseline', color=color_baseline, edgecolor='black', alpha=0.8)
rects2_a = ax2.bar(x + width/2, ai_service, width, label='PPO AI Agent', color=color_ai_profit, edgecolor='black', alpha=0.8)

ax2.set_title('Ποσοστό Εξυπηρέτησης Πελατών (Service Rate %)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Εξυπηρέτηση (%)', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(profiles, fontsize=11, rotation=15)
ax2.set_ylim(0, 115) 
ax2.legend()
add_full_labels(ax2, rects2_b, rects2_a, baseline_service, ai_service, format_pct)

fig2.tight_layout()
fig2.savefig('thesis_service_comparison.png', dpi=300, bbox_inches='tight')
print("Αποθηκεύτηκε: thesis_service_comparison.png")

# =====================================================================
# --- ΓΡΑΦΗΜΑ 3: ΧΡΟΝΟΣ ΑΝΑΜΟΝΗΣ ---
# =====================================================================
fig3, ax3 = plt.subplots(figsize=(14, 7))
rects3_b = ax3.bar(x - width/2, baseline_wait, width, label='Greedy Baseline', color=color_baseline, edgecolor='black', alpha=0.8)
rects3_a = ax3.bar(x + width/2, ai_wait, width, label='PPO AI Agent', color=color_ai_wait, edgecolor='black', alpha=0.8)

ax3.set_title('Μέσος Χρόνος Αναμονής Πελάτη', fontsize=14, fontweight='bold')
ax3.set_ylabel('Χρόνος (Λεπτά)', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels(profiles, fontsize=11, rotation=15)
ax3.set_ylim(0, max(max(baseline_wait), max(ai_wait)) * 1.15)
ax3.legend()
add_full_labels(ax3, rects3_b, rects3_a, baseline_wait, ai_wait, format_min, is_lower_better=True)

fig3.tight_layout()
fig3.savefig('thesis_wait_time_comparison.png', dpi=300, bbox_inches='tight')

# Εμφάνιση των παραθύρων
plt.show()