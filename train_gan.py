import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import networkx as nx
import matplotlib.pyplot as plt


def Net_Plot(G,gs):    
    pos=nx.kamada_kawai_layout(G)
    labels = {}
    for idx, node in enumerate(G.nodes()):
        labels[node] = idx

    fig_size = plt.rcParams["figure.figsize"]  
    fig_size[0] = gs; fig_size[1] = gs
    fig, ax = plt.subplots(1, 1, dpi=150)
    plt.rcParams["figure.figsize"] = fig_size  
    plt.title('')

    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='red', alpha=0.4)
    nx.draw_networkx_edges(G, pos, width=0.35)
    nx.draw_networkx_labels(G, pos, labels, font_size=9)
    plt.box(False)
    plt.show()


# Define the variable to store the loaded graphs
GraphsN3_SinCos2_Reps = []

# Iterate over each .gpickle file matching the pattern
for i in range(40, 50):
    # Load the graph from the .gpickle file
    file_path = f"SinCos/SinCos_NodeDel25_a[0]_b[1]_p[1]_Graphs_Pinp0.50_Pout0.50_Rp{i}.gpickle"

    with open(file_path, 'rb') as f:
        graph = pickle.load(f)

    GraphsN3_SinCos2_Reps.append(graph)

'''Build the GAN model'''

# Set the maximum number of nodes in the network
max_nodes = 50

# Function to generate a random network
def generate_random_network(num_nodes):
    # Generate a random adjacency matrix
    adj_matrix = np.random.randint(0, 2, size=(num_nodes, num_nodes))
    
    # Make the adjacency matrix symmetric
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    
    # Convert the adjacency matrix to a list of connections
    connections = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if adj_matrix[i, j] == 1:
                connections.append((i, j))
    
    return connections

# Convert a list of connections to an adjacency matrix
def connections_to_adj_matrix(connections, num_nodes):
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i, j in connections:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    return adj_matrix[:num_nodes, :num_nodes]

# Define the generator model
def build_generator(latent_dim, num_nodes):
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(num_nodes * num_nodes, activation='sigmoid'))
    model.add(Reshape((num_nodes, num_nodes)))
    return model

# Define the discriminator model
def build_discriminator(num_nodes):
    model = Sequential()
    model.add(Flatten(input_shape=(num_nodes, num_nodes)))
    model.add(Dense(512, activation='leaky_relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='leaky_relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Define the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Train the GAN
def train_gan(generator, discriminator, gan, X_train, epochs, batch_size, sample_interval):
    optimizer = Adam(0.0002, 0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    for epoch in range(epochs):
        # Train the discriminator
        idx = np.random.randint(0, len(X_train), batch_size)
        real_networks = [connections_to_adj_matrix(X_train[i], max_nodes) for i in idx]
        real_networks = np.array(real_networks)
        
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_networks = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_networks, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_networks, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        # Print progress
        print(f"Epoch {epoch}/{epochs}, D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}")
        
        # Generate and save sample networks
        if epoch % sample_interval == 0:
            sample_networks(generator, epoch)

# Generate and save sample networks
def sample_networks(generator, epoch):
    noise = np.random.normal(0, 1, (16, 100))
    generated_networks = generator.predict(noise)
    
    for i in range(16):
        # Convert the generated network to a list of connections
        connections = []
        for j in range(max_nodes):
            for k in range(j+1, max_nodes):
                if generated_networks[i, j, k] > 0.5:
                    connections.append((j, k))
        
        # Save the generated network
        # get the current directory
        cwd = os.getcwd()
        path = os.path.join(cwd, "generated_networks")
        np.save(f"{path}/generated_network_{epoch}_{i}.npy", connections)

# Function to plot the generated network
def plot_generated_network(epoch, sample_index):
    # Load the generated network
    generated_network = np.load(f"generated_network_{epoch}_{sample_index}.npy")
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges to the graph based on the generated network
    for connection in generated_network:
        G.add_edge(connection[0], connection[1])
    
    # Plot the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # Positions for all nodes
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=8, arrowsize=10)
    plt.title(f"Generated Network (Epoch {epoch}, Sample {sample_index})")
    plt.show()

# Function to plot the generated network
def plot_generated_network_var(connections, title="Generated Network"):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges to the graph based on the provided connections
    G.add_edges_from(connections)
    
    # Plot the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # Positions for all nodes
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=8, arrowsize=10)
    plt.title(title)
    plt.show()

X_train = []

temp = [i[-1].edges(data=True) for i in GraphsN3_SinCos2_Reps]
temp = [list(i) for i in temp]
for network in temp:
    temp_2 = []
    for node in network:
        temp_2.append((node[0], node[1]))
    X_train.append(temp_2)


# Example usage
num_nodes = 50
latent_dim = 100
epochs = 100
batch_size = 16
sample_interval = 10

# Generate random networks as training data
#X_train = [generate_random_network(num_nodes) for _ in range(1000)]
#X_train = [i[-1] for i in GraphsN3_SinCos2_Reps]


# Build the GAN
generator = build_generator(latent_dim, num_nodes)
discriminator = build_discriminator(num_nodes)
gan = build_gan(generator, discriminator)

# Train the GAN
train_gan(generator, discriminator, gan, X_train, epochs, batch_size, sample_interval)