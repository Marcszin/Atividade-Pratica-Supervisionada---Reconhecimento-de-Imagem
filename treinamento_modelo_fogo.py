import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

batch_size = 32
num_epochs = 20
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
])

dataset_path = 'dataset' 

if not os.path.exists(dataset_path):

    raise FileNotFoundError(
        "O diretório 'dataset' não foi encontrado. "
        "Crie-o com as subpastas 'fogo' e 'sem_fogo'."
    )

try:
    train_data = datasets.ImageFolder(root=dataset_path, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

except RuntimeError as e:
    print(f"\n[ERRO] Ocorreu um erro ao carregar o dataset: {e}")
    print("Verifique se as pastas 'fogo' e 'sem_fogo' dentro de 'dataset' contêm imagens.")

    if len(train_data.classes) < 2:
        print("Esperado 2 classes ('fogo' e 'sem_fogo'), verifique se as pastas existem.")
    exit()

class FireDetectionCNN(nn.Module):
    def __init__(self):
        super(FireDetectionCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.dropout = nn.Dropout(p=0.5) # Desliga 50% dos neurônios
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1) 

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        x = x.view(-1, 64 * 56 * 56)
        
        x = torch.relu(self.fc1(x))

        x = torch.sigmoid(self.fc2(x))
        return x

model = FireDetectionCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    print(f"Iniciando o treinamento em: {device.type.upper()}")
    model.train() # Coloca o modelo em modo de treinamento
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        progress_bar = tqdm(
            train_loader, 
            desc=f'Epoch [{epoch+1}/{num_epochs}]', 
            leave=True, 
            unit=' batch'
        )
        
        # O loop agora itera sobre a barra de progresso
        for i, (inputs, labels) in enumerate(progress_bar): 
            
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad() 
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward() 
            optimizer.step() 
            
            # --- MUDANÇA AQUI: Definição e Acumulação da Loss ---
            current_loss = loss.item() # <-- GARANTE QUE É DEFINIDA
            running_loss += current_loss
            
            # --- ATUALIZAÇÃO DA BARRA: Usa a variável que ACABOU de ser definida ---
            progress_bar.set_postfix({'Loss': f'{current_loss:.4f}'}) 
            # ----------------------------------------------------------------------
            

        avg_loss = running_loss/len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] concluída. Loss Média: {avg_loss:.4f}')
        
    print("\nTreinamento concluído!")

if __name__ == "__main__":
    try:
        train_model(model, train_loader, criterion, optimizer, num_epochs)
        torch.save(model.state_dict(), 'modelo_fogo.pth')
        print("Modelo salvo como 'modelo_fogo.pth'")
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        print("Verifique se a pasta 'dataset' com 'fogo' e 'sem_fogo' existe.")