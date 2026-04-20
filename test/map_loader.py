import osmnx as ox
import matplotlib.pyplot as plt

def main():
    # Συντεταγμένες κέντρου (π.χ. Λευκός Πύργος)
    center_point = (40.6264, 22.9484)
    
    # Ακτίνα σε μέτρα (4000 μέτρα = 4 χλμ)
    radius_meters = 5500 
    
    print(f"Κατέβασμα οδικού δικτύου σε ακτίνα {radius_meters}m από το κέντρο...")

    # Κατεβάζουμε τον γράφο γύρω από το συγκεκριμένο σημείο
    G = ox.graph_from_point(center_point, dist=radius_meters, network_type="drive")

    num_nodes = len(G.nodes)
    num_edges = len(G.edges)
    
    print("\nΕπιτυχία!")
    print(f"📍 Αριθμός Διασταυρώσεων (Nodes): {num_nodes}")
    print(f"🛣️ Αριθμός Δρόμων (Edges): {num_edges}")

    print("\nΣχεδιασμός χάρτη...")
    
    fig, ax = ox.plot_graph(
        G, 
        node_size=2,           # Λίγο πιο μεγάλες κουκκίδες τώρα που ο χάρτης είναι μικρότερος
        edge_linewidth=0.8,    
        bgcolor="#111111",     
        node_color="cyan",     
        edge_color="#555555",  
        show=True              
    )

if __name__ == "__main__":
    main()