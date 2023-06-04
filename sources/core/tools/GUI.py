import tkinter
import webbrowser
import ctypes
 
class GraphicalInterface(tkinter.Tk):
    
    def __init__(self, *args, **kwargs):
        
        tkinter.Tk.__init__(self, *args, **kwargs)
        self.call('tk', 'scaling', 2.0)
        
        # Adding a title to the window
        self.wm_title("PyXAI")
        self.create_menu_bar()
        # creating a frame and assigning it to container
        container = tkinter.Frame(self)
        # specifying the region where the frame is packed in root
        container.pack(side="top", fill="both", expand=True)

        # configuring the location of the container using grid
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        instance_bar = tkinter.Frame(container, borderwidth=1, relief="groove")
        instance_bar.pack(side="left", fill="both", expand=True)
        self.create_instance_bar(instance_bar)
        
        explanation_bar = tkinter.Frame(container, borderwidth=1, relief="groove")
        explanation_bar.pack(side="right", fill="both", expand=True)
        self.create_explanation_bar(explanation_bar)
        
    def create_instance_bar(self, instance_bar):
        label = tkinter.Label(instance_bar, text="Instance:", justify="left", anchor="w")
        label.pack(padx=0, pady=0)
        label.grid(sticky="w", row=0, columnspan=4)

        v1 = tkinter.StringVar()
        e1 = tkinter.Entry(instance_bar, textvariable = v1, state = 'readonly')
        v1.set('Name')
        e1.grid(row=1, column=1)

        v2 = tkinter.StringVar()
        e2 = tkinter.Entry(instance_bar, textvariable = v2, state = 'readonly')
        v2.set('Value')
        e2.grid(row=1, column=2) 


    def create_explanation_bar(self, explanation_bar):
        label = tkinter.Label(explanation_bar, text="Explanation:", justify="left", anchor="w")
        label.pack(padx=0, pady=2.5)
        label.grid(row=0)

    def create_menu_bar(self):
        menu_bar = tkinter.Menu(self)

        menu_file = tkinter.Menu(menu_bar, tearoff=0)
        menu_file.add_command(label="Save", command=self.save)
        menu_file.add_separator()
        menu_file.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="File", menu=menu_file)

        menu_help = tkinter.Menu(menu_bar, tearoff=0)
        menu_help.add_command(label="Documentation", command=self.documentation)
        menu_bar.add_cascade(label="Help", menu=menu_help)

        self.config(menu=menu_bar)

    def documentation(self):
        webbrowser.open_new("http://www.cril.univ-artois.fr/pyxai/documentation/")

    def save(self):
        print("Save clicked")


    