import matplotlib.pyplot as plt
import numpy as np

# --- 1. ΤΑ ΔΕΔΟΜΕΝΑ ---
labels = ['Baseline (Heuristic)', 'AI Fleet Manager (RL)']

profit = [176078.46, 182053.15]
served = [22534, 22884]
abandoned = [6586, 5888]
stars = [1.93, 1.77]

# Υπολογισμός Ποσοστών Διαφοράς
profit_diff = ((profit[1] - profit[0]) / profit[0]) * 100
served_diff = ((served[1] - served[0]) / served[0]) * 100
abandoned_diff = ((abandoned[1] - abandoned[0]) / abandoned[0]) * 100 
stars_diff = ((stars[1] - stars[0]) / stars[0]) * 100

# --- ΡΥΘΜΙΣΕΙΣ ΓΡΑΦΗΜΑΤΩΝ ---
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#FF9999', '#66B2FF'] # Κόκκινο για Baseline, Μπλε για AI

def create_plot(data, title, ylabel, filename, diff_percent, lower_is_better=False, is_float=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(labels, data, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    
    # Εμφάνιση των τιμών πάνω από τις μπάρες (με ή χωρίς δεκαδικά)
    for bar in bars:
        yval = bar.get_height()
        label_format = f'{yval:,.2f}' if is_float else f'{yval:,.0f}'
        ax.text(bar.get_x() + bar.get_width()/2, yval + (max(data)*0.01), 
                label_format, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Λογική Χρωμάτων (Πράσινο = Καλό, Κόκκινο = Κακό)
    if lower_is_better:
        box_color = 'green' if diff_percent < 0 else 'red'
    else:
        box_color = 'green' if diff_percent > 0 else 'red'
        
    arrow = '▲' if diff_percent > 0 else '▼'
    
    # Τοποθέτηση του κουτιού με το ποσοστό
    ax.text(1, data[1] / 2, f'{arrow} {abs(diff_percent):.2f}% \nΔιαφορά', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='white', 
            bbox=dict(facecolor=box_color, alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Δημιουργήθηκε: {filename}")
    plt.close()

# --- 2. ΔΗΜΙΟΥΡΓΙΑ ΤΩΝ 4 PLOTS ---

# 1. Καθαρό Κέρδος (Μεγαλύτερο = Καλύτερο)
create_plot(profit, 'Σύγκριση Καθαρού Κέρδους (Net Profit)', 'Ευρώ (€)', 
            'plot_1_profit.png', profit_diff)

# 2. Εξυπηρετημένοι Πελάτες (Μεγαλύτερο = Καλύτερο)
create_plot(served, 'Σύγκριση Εξυπηρετημένων Πελατών', 'Αριθμός Πελατών', 
            'plot_2_served.png', served_diff)

# 3. Εγκαταλείψεις (Μικρότερο = Καλύτερο)
create_plot(abandoned, 'Μείωση Εγκαταλείψεων (Χαμένοι Πελάτες)', 'Αριθμός Πελατών', 
            'plot_3_abandoned.png', abandoned_diff, lower_is_better=True)

# 4. Βαθμολογία Αστέρων (Μεγαλύτερο = Καλύτερο, αλλά εδώ πέσαμε, άρα θα βγει κόκκινο)
create_plot(stars, 'Ποιότητα Εξυπηρέτησης (Μέση Βαθμολογία)', 'Αστέρια (0-5)', 
            'plot_4_stars.png', stars_diff, is_float=True)