# Predição de Diabetes - Interface Gráfica Aprimorada
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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
        # Criar o explainer SHAP para o modelo
        explainer = shap.TreeExplainer(model)
        return model, scaler, explainer
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar modelo ou scaler: {e}")

model, scaler, explainer = load_model_and_scaler()



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
    root.geometry("480x490")
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
        
        # Calcular os valores SHAP para a explicação
        shap_values = explainer.shap_values(df_scaled)
        
        # Obter o valor base do modelo para a classe positiva (diabetes)
        base_value = explainer.expected_value[1]

        # Para explicação, usamos os valores SHAP para a classe positiva (índice 1 - diabetes)
        shap_values_diabetes = shap_values[0, :, 1]  # [instância, feature, classe]
        
        exibir_resultado(resultado[0], df_input, shap_values_diabetes, base_value)
    except Exception as e:
        messagebox.showerror("Erro", f"Verifique os dados inseridos.\n{e}")

def exibir_resultado(resultado, dados_originais, shap_valores, base_value):
    if resultado == 1:
        titulo = "Atenção: Possível diagnóstico de diabetes."
    else:
        titulo = "Sem indícios de diabetes."
    
    # Criar uma nova janela para a exibição do resultado e do gráfico
    resultado_window = tk.Toplevel()
    resultado_window.title("Resultado da Predição")
    resultado_window.geometry("900x600")  # Aumentar o tamanho da janela para melhor visualização
    resultado_window.configure(bg="#f0f4f8")
    
    # Centralizar a janela na tela
    # Pegar dimensões da tela
    screen_width = resultado_window.winfo_screenwidth()
    screen_height = resultado_window.winfo_screenheight()
    
    # Calcular posição
    x = (screen_width / 2) - (900 / 2)
    y = (screen_height / 2) - (600 / 2)
    
    # Definir geometria
    resultado_window.geometry(f"900x600+{int(x)}+{int(y)}")
    
    # Adicionar título
    tk.Label(
        resultado_window,
        text=titulo,
        font=("Arial", 14, "bold"),
        bg="#f0f4f8",
        fg="#007acc" if resultado == 0 else "#e74c3c"
    ).pack(pady=(20, 10))
    
    # Adicionar explicação
    tk.Label(
        resultado_window,
        text="O gráfico abaixo mostra os fatores que mais influenciaram este resultado:",
        font=("Arial", 12),
        bg="#f0f4f8"
    ).pack(pady=(5, 5))
    
    # Explicação das cores do gráfico SHAP
    tk.Label(
        resultado_window,
        text="Vermelho = aumenta chance de diabetes | Azul = diminui chance de diabetes",
        font=("Arial", 10, "italic"),
        bg="#f0f4f8",
        fg="#555555"
    ).pack(pady=(0, 5))
    
    # Explicação adicional para o gráfico de barras
    tk.Label(
        resultado_window,
        text="As características estão ordenadas por importância, com as mais impactantes no topo",
        font=("Arial", 10, "italic"),
        bg="#f0f4f8",
        fg="#555555"
    ).pack(pady=(0, 15))
    
    try:
        # Inicializar o JavaScript do SHAP (importante para os gráficos)
        shap.initjs()
        
        # Criar figura com tamanho maior para evitar sobreposição
        plt.figure(figsize=(12, 5))
        
        # Criar o gráfico de waterfall SHAP (alternativa ao force_plot que evita sobreposição)
        # Ordenar os valores SHAP para melhorar a legibilidade
        feature_names = dados_originais.columns.tolist()
        feature_importance = dict(zip(feature_names, abs(shap_valores)))
        sorted_idx = [i for i, _ in sorted(enumerate(abs(shap_valores)), key=lambda x: x[1], reverse=True)]
        
        # Criar o gráfico de barras SHAP em vez do force plot para evitar sobreposição
        plt.barh(
            [feature_names[i] for i in sorted_idx],
            [shap_valores[i] for i in sorted_idx],
            color=['#ff0051' if shap_valores[i] > 0 else '#008bfb' for i in sorted_idx]
        )
        plt.axvline(x=0, color='#333333', linestyle='-', alpha=0.3)
        plt.xlabel('Impacto no modelo (SHAP value)')
        plt.title('Contribuição de cada característica para a predição')
        
        # Definir a cor de fundo da figura como branco
        force_plot_fig = plt.gcf()
        force_plot_fig.set_facecolor('white')
        
        # Ajustar o layout para evitar cortes no gráfico
        plt.tight_layout(pad=2.0)
    except Exception as e:
        # Em caso de erro, criar uma figura simples com a mensagem de erro
        force_plot_fig = plt.figure(figsize=(10, 3))
        plt.text(0.5, 0.5, f"Erro ao criar gráfico SHAP: {e}", 
                horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
    
    # Criar o canvas para exibir o gráfico na janela do Tkinter
    canvas = FigureCanvasTkAgg(force_plot_fig, master=resultado_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    # Adicionar botão para fechar
    fechar_btn = ttk.Button(
        resultado_window,
        text="Fechar",
        command=resultado_window.destroy,
    )
    fechar_btn.pack(pady=(10, 20), ipadx=20, ipady=5)
    
    # Garantir que esta janela fique em foco
    resultado_window.transient(root)
    resultado_window.grab_set()
    root.wait_window(resultado_window)



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
    
    xai_label = tk.Label(
        root,
        text="Inclui explicação visual com tecnologia XAI",
        bg="#f0f4f8",
        fg="#555555",
        font=("Arial", 9, "italic")
    )
    xai_label.pack(side="bottom", pady=1)
    return entradas

entradas = criar_interface()
root.mainloop()