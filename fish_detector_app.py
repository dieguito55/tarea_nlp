"""
YOLO-Fish Detection App
Aplicación con interfaz gráfica para detectar peces en videos usando modelos YOLO
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading


class FishDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🐟 YOLO-Fish Detector Pro")
        
        # Obtener dimensiones de la pantalla y maximizar
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        self.root.state('zoomed')  # Maximizar en Windows
        self.root.configure(bg="#1a1a2e")
        
        # Variables
        self.model_path = ""
        self.config_path = ""
        self.video_path = ""
        self.net = None
        self.output_layers = None
        self.classes = ["Fish"]
        self.is_playing = False
        self.cap = None
        self.current_frame = None
        self.is_fullscreen = False
        self.total_frames = 0
        self.current_frame_number = 0
        self.video_slider_updating = False
        
        # Colores para las detecciones (colores vibrantes)
        self.colors = np.array([[0, 255, 255], [255, 0, 255], [0, 255, 0], [255, 255, 0]])
        
        # Bind para salir de fullscreen con ESC
        self.root.bind('<Escape>', self.exit_fullscreen)
        
        # Configurar estilos
        self.setup_styles()
        
        # Configuración de modelos disponibles
        self.models_config = {
            "DeepFish - YOLO-Fish-1": {
                "config": "models/trained_on_depfish/yolo-fish-1.cfg",
                "weights": "backup/depfish_yolo-fish-1.weights"
            },
            "DeepFish - YOLO-Fish-2": {
                "config": "models/trained_on_depfish/yolo-fish-2.cfg",
                "weights": "backup/depfish_yolo-fish-2.weights"
            },
            "DeepFish - YOLOv3": {
                "config": "models/trained_on_depfish/yolov3.cfg",
                "weights": "backup/depfish_yolov3.weights"
            },
            "OzFish - YOLO-Fish-1": {
                "config": "models/trained_on_ozfish/yolo-fish-1.cfg",
                "weights": "backup/ozfish_yolo-fish-1.weights"
            },
            "OzFish - YOLO-Fish-2": {
                "config": "models/trained_on_ozfish/yolo-fish-2.cfg",
                "weights": "backup/ozfish_yolo-fish-2.weights"
            },
            "OzFish - YOLOv3": {
                "config": "models/trained_on_ozfish/yolov3.cfg",
                "weights": "backup/ozfish_yolov3.weights"
            },
            "Merge - YOLO-Fish-1": {
                "config": "models/trained_on_merge/yolo-fish-1.cfg",
                "weights": "backup/merge_yolo-fish-1.weights"
            },
            "Merge - YOLO-Fish-2": {
                "config": "models/trained_on_merge/yolo-fish-2.cfg",
                "weights": "backup/merge_yolo-fish-2.weights"
            },
            "Merge - YOLOv3": {
                "config": "models/trained_on_merge/yolov3.cfg",
                "weights": "backup/merge_yolov3.weights"
            },
            "Merge - YOLOv4": {
                "config": "models/trained_on_merge/yolov4.cfg",
                "weights": "backup/merge_yolov4.weights"
            }
        }
        
        self.setup_ui()
    
    def setup_styles(self):
        """Configurar estilos modernos para la aplicación"""
        style = ttk.Style()
        
        # Tema base
        style.theme_use('clam')
        
        # Colores del tema
        bg_dark = "#1a1a2e"
        bg_medium = "#16213e"
        bg_light = "#0f3460"
        accent = "#00d4ff"
        accent_hover = "#00a8cc"
        text_color = "#ffffff"
        success = "#00ff88"
        warning = "#ff9500"
        
        # Configurar TFrame
        style.configure("TFrame", background=bg_dark)
        style.configure("Card.TFrame", background=bg_medium, relief="flat", borderwidth=2)
        style.configure("Header.TFrame", background=bg_light, relief="flat")
        
        # Configurar TLabel
        style.configure("TLabel",
                       background=bg_dark,
                       foreground=text_color,
                       font=("Segoe UI", 10))
        
        style.configure("Title.TLabel",
                       background=bg_light,
                       foreground=accent,
                       font=("Segoe UI", 24, "bold"))
        
        style.configure("Subtitle.TLabel",
                       background=bg_light,
                       foreground=text_color,
                       font=("Segoe UI", 11))
        
        style.configure("Header.TLabel",
                       background=bg_medium,
                       foreground=accent,
                       font=("Segoe UI", 11, "bold"))
        
        style.configure("Success.TLabel",
                       background=bg_dark,
                       foreground=success,
                       font=("Segoe UI", 10, "bold"))
        
        style.configure("Warning.TLabel",
                       background=bg_dark,
                       foreground=warning,
                       font=("Segoe UI", 10, "bold"))
        
        # Configurar TButton
        style.configure("TButton",
                       background=accent,
                       foreground=bg_dark,
                       borderwidth=0,
                       focuscolor="none",
                       font=("Segoe UI", 10, "bold"),
                       padding=10)
        
        style.map("TButton",
                 background=[("active", accent_hover), ("pressed", accent_hover)],
                 foreground=[("active", bg_dark)])
        
        style.configure("Accent.TButton",
                       background=accent,
                       foreground=bg_dark,
                       font=("Segoe UI", 11, "bold"),
                       padding=12)
        
        style.configure("Success.TButton",
                       background=success,
                       foreground=bg_dark,
                       font=("Segoe UI", 10, "bold"),
                       padding=10)
        
        # Configurar Combobox
        style.configure("TCombobox",
                       fieldbackground=bg_light,
                       background=accent,
                       foreground=text_color,
                       arrowcolor=accent,
                       borderwidth=0,
                       font=("Segoe UI", 10))
        
        # Configurar Progressbar
        style.configure("TProgressbar",
                       background=accent,
                       troughcolor=bg_light,
                       borderwidth=0,
                       thickness=20)
        
        # Configurar LabelFrame
        style.configure("TLabelframe",
                       background=bg_medium,
                       foreground=accent,
                       borderwidth=2,
                       relief="flat",
                       font=("Segoe UI", 11, "bold"))
        
        style.configure("TLabelframe.Label",
                       background=bg_medium,
                       foreground=accent,
                       font=("Segoe UI", 11, "bold"))
        
    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        
        # ==================== HEADER COMPACTO ====================
        header_frame = ttk.Frame(self.root, style="Header.TFrame", padding="10")
        header_frame.pack(fill=tk.X, side=tk.TOP)
        
        ttk.Label(
            header_frame,
            text="🐟 YOLO-Fish Detector Pro",
            font=("Segoe UI", 18, "bold"),
            foreground="#00d4ff",
            background="#0f3460"
        ).pack(side=tk.LEFT, padx=10)
        
        # ==================== PANEL DE CONTROL COMPACTO ====================
        control_panel = ttk.Frame(self.root, style="Card.TFrame", padding="10")
        control_panel.pack(fill=tk.X, padx=15, pady=(5, 0))
        
        # Primera fila: Modelo y Video en horizontal
        top_row = ttk.Frame(control_panel, style="TFrame")
        top_row.pack(fill=tk.X, pady=5)
        
        # Modelo
        ttk.Label(
            top_row,
            text="Modelo:",
            font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.model_var = tk.StringVar()
        model_dropdown = ttk.Combobox(
            top_row,
            textvariable=self.model_var,
            values=list(self.models_config.keys()),
            state="readonly",
            width=30,
            font=("Segoe UI", 9)
        )
        model_dropdown.pack(side=tk.LEFT, padx=5)
        model_dropdown.current(0)
        
        self.load_model_btn = ttk.Button(
            top_row,
            text="🔄 Cargar",
            command=self.load_model,
            style="TButton",
            width=10
        )
        self.load_model_btn.pack(side=tk.LEFT, padx=5)
        
        self.model_status_label = ttk.Label(
            top_row,
            text="⚠ No cargado",
            style="Warning.TLabel",
            font=("Segoe UI", 9)
        )
        self.model_status_label.pack(side=tk.LEFT, padx=10)
        
        # Spacer
        ttk.Frame(top_row, style="TFrame", width=20).pack(side=tk.LEFT)
        
        # Video
        self.select_video_btn = ttk.Button(
            top_row,
            text="📁 Video",
            command=self.select_video,
            state=tk.DISABLED,
            style="TButton",
            width=10
        )
        self.select_video_btn.pack(side=tk.LEFT, padx=5)
        
        self.video_label = ttk.Label(
            top_row,
            text="Ningún video",
            font=("Segoe UI", 9)
        )
        self.video_label.pack(side=tk.LEFT, padx=5)
        
        # ==================== CONTROLES DE REPRODUCCIÓN COMPACTOS ====================
        playback_row = ttk.Frame(control_panel, style="TFrame")
        playback_row.pack(pady=5)
        
        # Botón retroceder
        self.backward_btn = ttk.Button(
            playback_row,
            text="⏪ -10s",
            command=self.backward_video,
            state=tk.DISABLED,
            style="TButton",
            width=10
        )
        self.backward_btn.pack(side=tk.LEFT, padx=3)
        
        self.play_btn = ttk.Button(
            playback_row,
            text="▶ Play",
            command=self.play_video,
            state=tk.DISABLED,
            style="Success.TButton",
            width=10
        )
        self.play_btn.pack(side=tk.LEFT, padx=3)
        
        self.pause_btn = ttk.Button(
            playback_row,
            text="⏸ Pausa",
            command=self.pause_video,
            state=tk.DISABLED,
            style="TButton",
            width=10
        )
        self.pause_btn.pack(side=tk.LEFT, padx=3)
        
        self.stop_btn = ttk.Button(
            playback_row,
            text="⏹ Stop",
            command=self.stop_video,
            state=tk.DISABLED,
            style="TButton",
            width=10
        )
        self.stop_btn.pack(side=tk.LEFT, padx=3)
        
        # Botón adelantar
        self.forward_btn = ttk.Button(
            playback_row,
            text="⏩ +10s",
            command=self.forward_video,
            state=tk.DISABLED,
            style="TButton",
            width=10
        )
        self.forward_btn.pack(side=tk.LEFT, padx=3)
        
        # ==================== TIMELINE COMPACTA ====================
        timeline_panel = ttk.Frame(self.root, style="Card.TFrame", padding="8")
        timeline_panel.pack(fill=tk.X, padx=15, pady=(5, 5))
        
        progress_top = ttk.Frame(timeline_panel, style="TFrame")
        progress_top.pack(fill=tk.X, pady=(0, 5))
        
        self.time_label = ttk.Label(
            progress_top,
            text="00:00 / 00:00",
            font=("Segoe UI", 9)
        )
        self.time_label.pack(side=tk.LEFT, padx=5)
        
        self.frame_label = ttk.Label(
            progress_top,
            text="Frame: 0 / 0",
            font=("Segoe UI", 9)
        )
        self.frame_label.pack(side=tk.RIGHT, padx=5)
        
        # Slider para navegar en el video
        self.video_slider = tk.Scale(
            timeline_panel,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            command=self.on_slider_change,
            showvalue=False,
            bg="#0f3460",
            fg="#00d4ff",
            troughcolor="#16213e",
            highlightthickness=0,
            bd=0,
            sliderrelief=tk.FLAT,
            activebackground="#00a8cc"
        )
        self.video_slider.pack(fill=tk.X)
        
        # ==================== VISUALIZACIÓN DE VIDEO (MAXIMIZADA) ====================
        video_container = ttk.Frame(self.root, style="TFrame")
        video_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(5, 5))
        
        self.canvas = tk.Canvas(
            video_container,
            bg="#0a0a0a",
            highlightthickness=0,
            relief="flat"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Mensaje inicial en el canvas
        self.canvas.create_text(
            700, 300,
            text="🎬 Carga un modelo y selecciona un video para comenzar",
            font=("Segoe UI", 14),
            fill="#666666",
            tags="placeholder"
        )
        
        # Bind para actualizar canvas cuando cambie el tamaño
        self.canvas.bind('<Configure>', self.on_canvas_resize)
        
        # ==================== PANEL DE ESTADÍSTICAS COMPACTO ====================
        stats_panel = ttk.Frame(self.root, style="Card.TFrame", padding="8")
        stats_panel.pack(fill=tk.X, padx=15, pady=(5, 10))
        
        stats_container = ttk.Frame(stats_panel, style="TFrame")
        stats_container.pack()
        
        # Detecciones
        ttk.Label(
            stats_container,
            text="🐟",
            font=("Segoe UI", 11)
        ).pack(side=tk.LEFT, padx=(0, 3))
        
        self.detections_label = ttk.Label(
            stats_container,
            text="0",
            font=("Segoe UI", 11, "bold"),
            foreground="#00ff88"
        )
        self.detections_label.pack(side=tk.LEFT, padx=3)
        
        ttk.Label(
            stats_container,
            text="|",
            font=("Segoe UI", 11),
            foreground="#444444"
        ).pack(side=tk.LEFT, padx=10)
        
        # Confianza
        ttk.Label(
            stats_container,
            text="📊",
            font=("Segoe UI", 11)
        ).pack(side=tk.LEFT, padx=3)
        
        self.confidence_label = ttk.Label(
            stats_container,
            text="0%",
            font=("Segoe UI", 11, "bold"),
            foreground="#00d4ff"
        )
        self.confidence_label.pack(side=tk.LEFT, padx=3)
        
        ttk.Label(
            stats_container,
            text="|",
            font=("Segoe UI", 11),
            foreground="#444444"
        ).pack(side=tk.LEFT, padx=10)
        
        # FPS
        ttk.Label(
            stats_container,
            text="⚡",
            font=("Segoe UI", 11)
        ).pack(side=tk.LEFT, padx=3)
        
        self.fps_label = ttk.Label(
            stats_container,
            text="0 fps",
            font=("Segoe UI", 11, "bold"),
            foreground="#ff9500"
        )
        self.fps_label.pack(side=tk.LEFT, padx=3)
        
    def load_model(self):
        """Cargar el modelo YOLO seleccionado"""
        selected_model = self.model_var.get()
        
        if not selected_model:
            messagebox.showerror("Error", "Por favor seleccione un modelo")
            return
        
        model_info = self.models_config[selected_model]
        config_path = model_info["config"]
        weights_path = model_info["weights"]
        
        # Verificar que existen los archivos
        if not os.path.exists(config_path):
            messagebox.showerror(
                "Error", 
                f"No se encontró el archivo de configuración:\n{config_path}"
            )
            return
        
        if not os.path.exists(weights_path):
            messagebox.showwarning(
                "Archivo de pesos no encontrado",
                f"No se encontró el archivo de pesos:\n{weights_path}\n\n"
                f"Por favor descárgalo del repositorio y colócalo en la carpeta 'backup/'"
            )
            return
        
        try:
            # Cargar el modelo
            self.net = cv2.dnn.readNet(weights_path, config_path)
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            self.config_path = config_path
            self.model_path = weights_path
            
            # Actualizar UI
            self.model_status_label.config(
                text="✅ Cargado",
                style="Success.TLabel"
            )
            self.select_video_btn.config(state=tk.NORMAL)
            
            messagebox.showinfo(
                "Éxito",
                f"✅ Modelo cargado correctamente\n\n{selected_model}"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar el modelo:\n{str(e)}")
    
    def select_video(self):
        """Seleccionar archivo de video"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[
                ("Archivos de video", "*.mp4 *.avi *.mov *.mkv *.flv"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if file_path:
            self.video_path = file_path
            filename = os.path.basename(file_path)
            self.video_label.config(text=f"📹 {filename}")
            self.play_btn.config(state=tk.NORMAL)
            self.canvas.delete("placeholder")
            
            # Abrir video para obtener información
            temp_cap = cv2.VideoCapture(file_path)
            self.total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = temp_cap.get(cv2.CAP_PROP_FPS)
            duration = self.total_frames / fps if fps > 0 else 0
            temp_cap.release()
            
            # Actualizar slider
            self.video_slider.config(to=self.total_frames - 1)
            
            # Mostrar duración
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            self.time_label.config(text=f"00:00 / {minutes:02d}:{seconds:02d}")
            self.frame_label.config(text=f"Frame: 0 / {self.total_frames}")
            
    def play_video(self):
        """Reproducir video con detección"""
        if not self.video_path:
            messagebox.showwarning("Advertencia", "Por favor seleccione un video")
            return
        
        if self.net is None:
            messagebox.showwarning("Advertencia", "Por favor cargue un modelo primero")
            return
        
        self.is_playing = True
        self.play_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.forward_btn.config(state=tk.NORMAL)
        self.backward_btn.config(state=tk.NORMAL)
        
        # Entrar en modo fullscreen
        self.enter_fullscreen()
        
        # Iniciar procesamiento en un hilo separado
        thread = threading.Thread(target=self.process_video, daemon=True)
        thread.start()
    
    def pause_video(self):
        """Pausar reproducción"""
        self.is_playing = False
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        
        # Salir de fullscreen al pausar
        self.exit_fullscreen()
    
    def forward_video(self):
        """Adelantar 10 segundos"""
        if self.cap is not None and self.cap.isOpened():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            skip_frames = int(fps * 10)  # 10 segundos
            new_frame = min(self.current_frame_number + skip_frames, self.total_frames - 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            self.current_frame_number = new_frame
            self.update_progress_display()
    
    def backward_video(self):
        """Retroceder 10 segundos"""
        if self.cap is not None and self.cap.isOpened():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            skip_frames = int(fps * 10)  # 10 segundos
            new_frame = max(self.current_frame_number - skip_frames, 0)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            self.current_frame_number = new_frame
            self.update_progress_display()
    
    def on_slider_change(self, value):
        """Manejar cambio en el slider"""
        if self.video_slider_updating:
            return
        
        if self.cap is not None and self.cap.isOpened():
            frame_number = int(float(value))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame_number = frame_number
            self.update_progress_display()
    
    def on_canvas_resize(self, event):
        """Manejar redimensionamiento del canvas"""
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            self.display_frame(self.current_frame)
    
    def update_progress_display(self):
        """Actualizar visualización del progreso"""
        if self.total_frames > 0:
            fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30
            
            # Actualizar slider
            self.video_slider_updating = True
            self.video_slider.set(self.current_frame_number)
            self.video_slider_updating = False
            
            # Actualizar tiempo
            current_time = self.current_frame_number / fps if fps > 0 else 0
            total_time = self.total_frames / fps if fps > 0 else 0
            
            curr_min = int(current_time // 60)
            curr_sec = int(current_time % 60)
            total_min = int(total_time // 60)
            total_sec = int(total_time % 60)
            
            self.time_label.config(text=f"{curr_min:02d}:{curr_sec:02d} / {total_min:02d}:{total_sec:02d}")
            self.frame_label.config(text=f"Frame: {self.current_frame_number} / {self.total_frames}")
    
    def stop_video(self):
        """Detener reproducción"""
        self.is_playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Salir de fullscreen
        self.exit_fullscreen()
        
        self.canvas.delete("all")
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.forward_btn.config(state=tk.DISABLED)
        self.backward_btn.config(state=tk.DISABLED)
        self.video_slider.set(0)
        self.frame_label.config(text="Frame: 0 / 0")
        self.time_label.config(text="00:00 / 00:00")
        self.detections_label.config(text="0")
        self.confidence_label.config(text="0%")
        self.fps_label.config(text="0")
        self.current_frame_number = 0
        
        # Restaurar placeholder
        if self.video_path:
            # Obtener dimensiones actuales del canvas
            self.canvas.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            self.canvas.create_text(
                canvas_width // 2, canvas_height // 2,
                text="▶ Video detenido. Presiona Reproducir para continuar",
                font=("Segoe UI", 14),
                fill="#666666",
                tags="placeholder"
            )
    
    def process_video(self):
        """Procesar video frame por frame"""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir el video")
            return
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_delay = int(1000 / fps) if fps > 0 else 33
        
        # Si hay un frame number guardado, ir a esa posición
        if self.current_frame_number > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        
        import time
        
        while self.is_playing and self.cap.isOpened():
            start_time = time.time()
            
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Realizar detección
            detected_frame, detections, avg_confidence = self.detect_fish(frame)
            
            # Guardar frame actual
            self.current_frame = detected_frame.copy()
            
            # Mostrar frame procesado
            self.display_frame(detected_frame)
            
            # Actualizar progreso
            self.update_progress_display()
            
            # Calcular FPS real
            processing_time = time.time() - start_time
            current_fps = 1 / processing_time if processing_time > 0 else 0
            
            # Actualizar estadísticas
            self.detections_label.config(text=str(detections))
            self.confidence_label.config(text=f"{avg_confidence:.1f}%")
            self.fps_label.config(text=f"{current_fps:.1f} fps")
            
            # Esperar para mantener FPS
            remaining_time = max(1, frame_delay - int(processing_time * 1000))
            self.root.update()
            time.sleep(remaining_time / 1000.0)
        
        # Finalizar
        self.cap.release()
        self.is_playing = False
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        
        if self.current_frame_number >= self.total_frames - 1:
            messagebox.showinfo("Finalizado", "El video ha terminado de reproducirse")
            self.exit_fullscreen()
    
    def enter_fullscreen(self):
        """Entrar en modo pantalla completa"""
        if not self.is_fullscreen:
            self.is_fullscreen = True
            self.root.attributes('-fullscreen', True)
            self.root.config(cursor="none")  # Ocultar cursor
    
    def exit_fullscreen(self, event=None):
        """Salir del modo pantalla completa"""
        if self.is_fullscreen:
            self.is_fullscreen = False
            self.root.attributes('-fullscreen', False)
            self.root.config(cursor="")  # Restaurar cursor
    
    def detect_fish(self, frame):
        """Detectar peces en el frame usando YOLO"""
        height, width, channels = frame.shape
        
        # Preparar imagen para YOLO
        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
        )
        
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        # Información de detecciones
        class_ids = []
        confidences = []
        boxes = []
        
        # Procesar detecciones
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:  # Umbral de confianza
                    # Coordenadas del cuadro delimitador
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Coordenadas del rectángulo
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Aplicar Non-Maximum Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Dibujar detecciones
        detection_count = 0
        total_confidence = 0
        
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = tuple(map(int, self.colors[i % len(self.colors)]))
                
                # Dibujar rectángulo con grosor mayor
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Crear fondo para la etiqueta
                label_text = f"{label}: {confidence:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Fondo de la etiqueta
                cv2.rectangle(
                    frame,
                    (x, y - text_height - 10),
                    (x + text_width + 10, y),
                    color,
                    -1
                )
                
                # Texto de la etiqueta en negro para mejor contraste
                cv2.putText(
                    frame,
                    label_text,
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )
                
                detection_count += 1
                total_confidence += confidence * 100
        
        avg_confidence = total_confidence / detection_count if detection_count > 0 else 0
        
        return frame, detection_count, avg_confidence
    
    def display_frame(self, frame):
        """Mostrar frame en el canvas"""
        # Convertir de BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Actualizar canvas y obtener dimensiones reales
        self.canvas.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Calcular ratio para mantener aspecto
            frame_height, frame_width = frame_rgb.shape[:2]
            ratio = min(canvas_width / frame_width, canvas_height / frame_height)
            
            new_width = int(frame_width * ratio)
            new_height = int(frame_height * ratio)
            
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Convertir a ImageTk
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Limpiar canvas
            self.canvas.delete("all")
            
            # Si está en fullscreen, agregar fondo negro
            if self.is_fullscreen:
                self.canvas.create_rectangle(
                    0, 0, canvas_width, canvas_height,
                    fill="#000000", outline="", tags="background"
                )
            
            # Centrar imagen perfectamente
            x_pos = canvas_width // 2
            y_pos = canvas_height // 2
            
            self.canvas.create_image(
                x_pos, y_pos,
                anchor=tk.CENTER, image=imgtk,
                tags="video"
            )
            self.canvas.image = imgtk
            
            # Agregar indicador de fullscreen
            if self.is_fullscreen:
                self.canvas.create_text(
                    canvas_width - 20, 20,
                    text="Presiona ESC para salir de pantalla completa",
                    font=("Segoe UI", 10),
                    fill="#ffffff",
                    anchor=tk.NE,
                    tags="hint"
                )


def main():
    root = tk.Tk()
    app = FishDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
