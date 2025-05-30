# Predição de Diabetes - Interface Gráfica Aprimorada
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import pandas as pd
import joblib
import os


MODEL_FILENAME = 'random_forest_diabetes.pkl'
SCALER_FILENAME = 'scaler_diabetes.pkl'

FIELDS = [
    "Gênero",
    "Idade",
    "Hipertensão",
    "Doença Cardíaca",
    "Histórico de Fumo",
    "Peso (kg)",
    "Altura (cm)",
    "Nível de HbA1c",
    "Glicose no Sangue"
]

def load_model_and_scaler():
    """Carrega o modelo e o scaler do disco."""
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
    scaler_path = os.path.join(os.path.dirname(__file__), SCALER_FILENAME)
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar modelo ou scaler: {e}")

model, scaler = load_model_and_scaler()



def get_field_options():
    return {
        "Gênero": ["Feminino", "Masculino"],
        "Hipertensão": ["Não", "Sim"],
        "Doença Cardíaca": ["Não", "Sim"],
        "Histórico de Fumo": [
            "Nunca fumou", "Sem informação", "Fumante atual", "Ex-fumante", "Já fumou", "Não é fumante atual"
        ]
    }

def create_main_window():
    root = tk.Tk()
    root.title("Predição de Diabetes")
    root.configure(bg="#f0f4f8")
    root.geometry("470x470")
    return root

def create_field_vars():
    options = get_field_options()
    return {
        "Gênero": tk.StringVar(value=options["Gênero"][0]),
        "Hipertensão": tk.StringVar(value=options["Hipertensão"][0]),
        "Doença Cardíaca": tk.StringVar(value=options["Doença Cardíaca"][0]),
        "Histórico de Fumo": tk.StringVar(value=options["Histórico de Fumo"][0])
    }

root = create_main_window()
field_vars = create_field_vars()
smoking_options = get_field_options()["Histórico de Fumo"]


def calcular_imc(peso, altura_cm):
    """Calcula o IMC a partir do peso (kg) e altura (cm)."""
    altura_m = altura_cm / 100.0
    return peso / (altura_m ** 2)

def obter_dados_usuario(entradas):
    """Obtém e processa os dados inseridos pelo usuário."""
    dados = []
    peso = None
    altura = None
    for i, campo in enumerate(FIELDS):
        if campo == "Gênero":
            valor = 1 if field_vars["Gênero"].get() == "Masculino" else 0
            dados.append(valor)
        elif campo == "Hipertensão":
            valor = 1 if field_vars["Hipertensão"].get() == "Sim" else 0
            dados.append(valor)
        elif campo == "Doença Cardíaca":
            valor = 1 if field_vars["Doença Cardíaca"].get() == "Sim" else 0
            dados.append(valor)
        elif campo == "Histórico de Fumo":
            valor = smoking_options.index(field_vars["Histórico de Fumo"].get())
            dados.append(valor)
        elif campo == "Peso (kg)":
            peso = float(entradas[i].get())
        elif campo == "Altura (cm)":
            altura = float(entradas[i].get())
        else:
            valor = float(entradas[i].get())
            dados.append(valor)
    if peso is None or altura is None:
        raise ValueError("Peso e altura devem ser informados.")
    imc = calcular_imc(peso, altura)
    dados.insert(5, imc)
    return dados

def prever_diabetes():
    try:
        dados = obter_dados_usuario(entradas)
        colunas = [
            "gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"
        ]
        df_input = pd.DataFrame([dados], columns=colunas)
        dados_scaled = scaler.transform(df_input)
        df_scaled = pd.DataFrame(dados_scaled, columns=colunas)
        resultado = model.predict(df_scaled)
        exibir_resultado(resultado[0])
    except Exception as e:
        messagebox.showerror("Erro", f"Verifique os dados inseridos.\n{e}")

def exibir_resultado(resultado):
    if resultado == 1:
        msg = "Atenção: Possível diagnóstico de diabetes."
    else:
        msg = "Sem indícios de diabetes."
    messagebox.showinfo("Resultado", msg)



def criar_interface():
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", background="#f0f4f8", font=("Arial", 11))
    style.configure("TButton", font=("Arial", 12, "bold"), foreground="#fff", background="#007acc")
    style.map("TButton", background=[("active", "#005f99")])

    main_frame = tk.Frame(root, bg="#f0f4f8")
    main_frame.pack(padx=20, pady=20)

    entradas = []
    options = get_field_options()
    for i, campo in enumerate(FIELDS):
        frame = tk.Frame(main_frame, bg="#f0f4f8")
        label = ttk.Label(frame, text=campo)
        label.pack(side="left", padx=(0, 8))
        if campo in options:
            combo = ttk.Combobox(
                frame,
                textvariable=field_vars[campo],
                values=options[campo],
                width=22 if campo == "Histórico de Fumo" else 12,
                state="readonly"
            )
            combo.pack(side="left")
            entradas.append(None)
        else:
            entry = ttk.Entry(frame, width=14)
            entry.pack(side="left")
            entradas.append(entry)
        frame.pack(pady=7, anchor="w")

    btn = ttk.Button(root, text="Verificar Diabetes", command=lambda: prever_diabetes())
    btn.pack(pady=18, ipadx=10, ipady=4)

    tip_label = tk.Label(
        root,
        text="Preencha todos os campos antes de verificar.",
        bg="#f0f4f8",
        fg="#007acc",
        font=("Arial", 10, "italic")
    )
    tip_label.pack(side="bottom", pady=2)
    return entradas

entradas = criar_interface()
root.mainloop()