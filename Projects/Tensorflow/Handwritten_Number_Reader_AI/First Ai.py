"""
Importazione dei moduli necessari
"""
import os
import torch
import torch.nn as nn          # Modulo per le reti neurali
import torch.optim as optim    # Modulo per gli algoritmi di ottimizzazione
import torchvision             # Modulo per i dataset e le trasformazioni sulle immagini
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

current_dir = os.getcwd()
Dataset_dir = os.path.join(current_dir, 'Dataset')

print(Dataset_dir)

input("Premi INVIO per continuare...")

"""
Preparazione del Dataset
"""

transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root=Dataset_dir, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(root=Dataset_dir, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

""" 
Definizione della Rete Neurale come oggetto di una classe
"""
class ReteNeurale(nn.Module):
    def __init__(self):
        super(ReteNeurale, self).__init__()
        self.flatten = nn.Flatten()                           # Appiattisce l'immagine 28x28 in un vettore di 784 elementi
        self.fc1 = nn.Linear(28 * 28, 128)                    # Primo strato completamente connesso
        self.relu1 = nn.ReLU()                                # Funzione di attivazione ReLU
        self.fc2 = nn.Linear(128, 64)                         # Secondo strato completamente connesso
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 10)                       # Strato di output per le 10 classi

    def forward(self, x):
        x = self.flatten(x)           # Appiattimento dell'input
        x = self.relu1(self.fc1(x))   # Primo strato + ReLU
        x = self.relu2(self.fc2(x))   # Secondo strato + ReLU
        x = self.output(x)            # Strato di output (senza funzione di attivazione)
        return x


# Creazione dell'istanza della rete neurale chiamata modello
modello = ReteNeurale()

# Definizione della funzione di perdita guida il processo di apprendimento, indicando al modello quanto migliorare durante l'addestramento.
criterio = nn.CrossEntropyLoss()     

# Definizione dell'ottimizzatore
ottimizzatore = optim.Adam(modello.parameters(), lr=0.001)  # Algoritmo di ottimizzazione Adam flessibile ; Il lr determina quanto grandi sono i passi che l'ottimizzatore compie per minimizzare la funzione di perdita.


num_epoche = 3
# Numero di epoche per il training

"""
Allenamento del Modello
"""
for epoca in range(num_epoche):
    running_loss = 0.0
    for immagine, etichetta in trainloader:
        ottimizzatore.zero_grad()                # Azzeramento dei gradienti
        
        output = modello(immagine)               # Forward pass
        perdita = criterio(output, etichetta)    # Calcolo della perdita
        perdita.backward()                       # Backpropagation
        ottimizzatore.step()                     # Aggiornamento dei pesi
        
        running_loss += perdita.item()

    perdita_media = running_loss / len(trainloader)
    print(f"Epoca {epoca+1}/{num_epoche}, Perdita Media: {perdita_media:.4f}")

"""
Valutazione del Modello
"""
corrette = 0
totali = 0
modello.eval()  # Imposta il modello in modalità valutazione

with torch.no_grad():  # Disabilita il calcolo dei gradienti
    for immagine, etichetta in testloader:
        output = modello(immagine)                      # Forward pass
        _, predizioni = torch.max(output, 1)            # Ottieni la classe con la massima probabilità
        totali += etichetta.size(0)
        corrette += (predizioni == etichetta).sum().item()

accuratezza = 100 * corrette / totali
print(f'Accuratezza sul set di test: {accuratezza:.2f}%')

"""
Utilizzo del Modello
"""

loop = True

while loop == True: 
    
    int_imp = int(input("Inserisci un numero da 0 a 9999: "))
    
    # Prendi un'immagine dal dataset di test
    immagine_singola, etichetta_reale = testset[int_imp]

    # Visualizza l'immagine
    plt.imshow(immagine_singola.squeeze(), cmap='gray')
    plt.show()

    # Preparazione dell'immagine per il modello
    modello.eval()
    with torch.no_grad():
        immagine_input = immagine_singola.unsqueeze(0)    # Aggiungi una dimensione per il batch
        output = modello(immagine_input)
        _, predizione = torch.max(output, 1)
        

    print(f'Numero reale: {etichetta_reale}')
    print(f'Predizione predetto dal modello: {predizione.item()}')
