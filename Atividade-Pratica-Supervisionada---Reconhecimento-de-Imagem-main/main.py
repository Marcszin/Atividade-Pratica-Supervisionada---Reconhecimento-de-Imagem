import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from treinamento_modelo_fogo import FireDetectionCNN
from tkinter import Tk, filedialog
import os 

#  Configurações de Exibição 
NEW_WIDTH = 1000 
NEW_HEIGHT = 600  
WINDOW_NAME = 'Detector de Fogo - APS (Pressione Q para Sair)'

#  (Laranja/Vermelho) 
lower_orange = (5, 150, 150)
upper_orange = (15, 255, 255)
lower_red_light = (0, 150, 150)
upper_red_light = (10, 255, 255)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


try:
    model = FireDetectionCNN().to(device)

    model.load_state_dict(torch.load('modelo_fogo.pth', map_location=device))
    model.eval()
except FileNotFoundError:
    print("[ERRO] O arquivo 'modelo_fogo.pth' não foi encontrado.")
    print("Execute 'treinamento_modelo_fogo.py' primeiro para gerar o modelo.")
    exit()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def detect_fire(frame):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tensor = transform(frame_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(frame_tensor)
        prediction = torch.sigmoid(output)

    confidence = prediction.item()
    print(f'Confiança do modelo: {confidence:.4f}') 
    
    threshold = 0.51
    return confidence > threshold

def check_fire_color(frame):

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_orange = cv2.inRange(hsv_frame, lower_orange, upper_orange)
    mask_red_light = cv2.inRange(hsv_frame, lower_red_light, upper_red_light)
    combined_mask = cv2.bitwise_or(mask_orange, mask_red_light)
    area = cv2.countNonZero(combined_mask)
    
    min_area_threshold = 300 
    return area > min_area_threshold, combined_mask

def detect_fire_in_video(frame):

    has_color, mask = check_fire_color(frame)
    if has_color and detect_fire(frame):
        return True, mask
    return False, None


def play_video(video_files):
    if isinstance(video_files, str):
        video_files = [video_files]
        
    if not video_files or not video_files[0]:
        return 

    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            print(f"Erro ao abrir o vídeo {video_file}!")
            continue

        print(f"\n Reproduzindo vídeo: {os.path.basename(video_file)}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break 

            frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT), interpolation=cv2.INTER_AREA)

            fire_detected, mask = detect_fire_in_video(frame)

            if fire_detected and mask is not None:

                frame_green = frame.copy()
                frame_green[mask > 0] = (0, 255, 0)
                
                frame = cv2.addWeighted(frame, 0.40, frame_green, 0.60, 0) 

                cv2.putText(frame, "FOGO DETECTADO", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (198, 129, 7), 3, cv2.LINE_AA)

            cv2.imshow('Detector de Fogo - APS (Pressione Q para Sair)', frame)

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()

    cv2.destroyAllWindows()

#Selecione o Arquivo de Vídeo
def select_video_file():

    root = Tk()
    root.withdraw() 

    file_path = filedialog.askopenfilename(
        title="Selecione o Arquivo de Vídeo para Detecção de Fogo",
        initialdir=".",
        filetypes=(
            ("Arquivos de Vídeo", "*.mp4 *.avi *.mov *.mkv"), 
            ("Todos os Arquivos", "*.*")
        )
    )
    
    return file_path

if __name__ == "__main__":
    caminho_video = select_video_file()
    
    if caminho_video:
        play_video(caminho_video)
    else:
        print("\n Operação cancelada pelo usuário.")