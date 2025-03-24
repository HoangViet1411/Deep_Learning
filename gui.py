import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2  # Import cv2
import main
import collect_imgs
import create_dataset
import train_classifier
import threading

class HandSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Sign Detection")

        # Create menu bar
        self.menu_bar = tk.Menu(root)
        root.config(menu=self.menu_bar)

        # Add "Hướng dẫn sử dụng" menu
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Hướng dẫn sử dụng", menu=self.help_menu)
        self.help_menu.add_command(label="Hướng dẫn", command=self.show_guide)

        # Add "About" menu
        self.about_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="About", menu=self.about_menu)
        self.about_menu.add_command(label="About", command=self.show_about)

        # Create a frame for the tabs
        self.tab_frame = ttk.Frame(root)
        self.tab_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Create a notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create tabs
        self.create_collect_img_tab()
        self.create_create_dataset_tab()
        self.create_train_classifier_tab()
        self.create_main_tab()  # Add main.py tab

        # Initialize canvas
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        # Initialize text area
        self.text_area = tk.Text(root, height=4, font=("Helvetica", 16))
        self.text_area.pack(pady=20)

        self.running = False  # Variable to track if the process is running

    def create_collect_img_tab(self):
        collect_img_tab = ttk.Frame(self.notebook)
        self.notebook.add(collect_img_tab, text="Tạo dữ liệu")

        # Add content to the collect_img_tab
        collect_img_label = ttk.Label(collect_img_tab, text="Tạo Dữ Liệu", font=("Helvetica", 16))
        collect_img_label.pack(pady=20)

        # Add buttons to run and stop the collect_img script
        collect_img_button = ttk.Button(collect_img_tab, text="Run", command=self.run_collect_img)
        collect_img_button.pack(pady=10)
        stop_collect_img_button = ttk.Button(collect_img_tab, text="Stop", command=self.stop_collect_img)
        stop_collect_img_button.pack(pady=10)
        self.folder_entry = ttk.Entry(collect_img_tab)
        self.folder_entry.pack(pady=10)

    def create_create_dataset_tab(self):
        create_dataset_tab = ttk.Frame(self.notebook)
        self.notebook.add(create_dataset_tab, text="Tạo Dataset")

        # Add content to the create_dataset_tab
        create_dataset_label = ttk.Label(create_dataset_tab, text="Tạo dataset", font=("Helvetica", 16))
        create_dataset_label.pack(pady=20)

        # Add a button to run the create_dataset script
        create_dataset_button = ttk.Button(create_dataset_tab, text="Run", command=self.run_create_dataset)
        create_dataset_button.pack(pady=10)

    def create_train_classifier_tab(self):
        train_classifier_tab = ttk.Frame(self.notebook)
        self.notebook.add(train_classifier_tab, text="Train model")

        # Add content to the train_classifier_tab
        train_classifier_label = ttk.Label(train_classifier_tab, text="Train Model", font=("Helvetica", 16))
        train_classifier_label.pack(pady=20)

        # Add a button to run the train_classifier script
        train_classifier_button = ttk.Button(train_classifier_tab, text="Run", command=self.run_train_classifier)
        train_classifier_button.pack(pady=10)

    def create_main_tab(self):
        main_tab = ttk.Frame(self.notebook)
        self.notebook.add(main_tab, text="Model Thủ Ngữ")

        # Add content to the main_tab
        main_label = ttk.Label(main_tab, text="Model Thủ Ngữ", font=("Helvetica", 16))
        main_label.pack(pady=20)

        # Add buttons to run and stop the main script
        main_button = ttk.Button(main_tab, text="Run", command=self.run_main)
        main_button.pack(pady=10)
        stop_main_button = ttk.Button(main_tab, text="Stop", command=self.stop_main)
        stop_main_button.pack(pady=10)

    def run_create_dataset(self):
        create_dataset.main()

    def run_train_classifier(self):
        train_classifier.main()

    def run_main(self):
        self.running = True
        main.cap = cv2.VideoCapture(0)  # Reinitialize the camera
        self.update_main_frame()

    def stop_main(self):
        self.running = False
        self.canvas.delete("all")  # Clear the canvas
        main.cap.release()
        cv2.destroyAllWindows()

    def run_collect_img(self):
        folder_name = self.folder_entry.get()  # Lấy tên thư mục từ entry
        if not folder_name:
            tk.messagebox.showerror("Lỗi", "Vui lòng nhập tên thư mục.")
            return

        self.running = True
        collect_imgs.cap = cv2.VideoCapture(0)
        threading.Thread(target=lambda: collect_imgs.collect_images(folder_name)).start() #thay đổi để nhận tham số folder name

    def stop_collect_img(self):
        self.running = False
        self.canvas.delete("all")  
        collect_imgs.release_resources()

    def update_main_frame(self):
        if not self.running:
            return
        frame, word = main.process_frame()
        if frame is not None:
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert(tk.END, word)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

        self.root.after(10, self.update_main_frame)
        main.labels_dict = main.get_labels_from_pickle() #cập nhật label từ file pickle.

    def update_collect_img_frame(self):
        if not self.running:
            return
        ret, frame = collect_imgs.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

            # Handle key press events for 'Q' to start capturing images
            self.root.bind('<KeyPress>', self.on_key_press_collect_img)

        self.root.after(10, self.update_collect_img_frame)

    def on_key_press_collect_img(self, event):
        if event.char == 'q':
            collect_imgs.collect_images(0)  # Adjust the class index as needed

    def on_key_press(self, event):
        if event.char == 'c':
            self.text_area.delete("1.0", tk.END)
            main.word = ""
        elif event.keysym == 'BackSpace':
            current_text = self.text_area.get("1.0", tk.END).strip()
            if current_text:
                new_text = current_text[:-1]
                self.text_area.delete("1.0", tk.END)
                self.text_area.insert(tk.END, new_text)
                main.word = new_text


    def on_closing(self):
        main.release_resources()
        self.root.destroy()

    def show_guide(self):
        guide_window = tk.Toplevel(self.root)
        guide_window.title("Hướng dẫn sử dụng")
        guide_label = ttk.Label(guide_window, text="Hướng dẫn sử dụng ứng dụng...", font=("Helvetica", 12))
        guide_label.pack(padx=20, pady=20)

    def show_about(self):
        about_window = tk.Toplevel(self.root)
        about_window.title("About")
        about_label = ttk.Label(about_window, text="Được tạo bởi Nguyễn Lê Anh Toàn và Phạm Đình Gia Huy", font=("Helvetica", 12))
        about_label.pack(padx=20, pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = HandSignApp(root)
    root.bind('<KeyPress>', app.on_key_press)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
