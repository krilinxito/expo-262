# Expo-DAT-262
Repositorio para los documentos para la expo
## 📁 Estructura General

```
Expo/
├── Proy3/
│   ├── Informe_Proy_3.md
│   ├── img/
│   └── code/
│       ├── requirements.txt
│       └── venv/  ← (No se sube al repo)
│
├── Proy13/
│   ├── Informe_Proy_13.md
│   ├── img/
│   └── code/
│       ├── requirements.txt
│       └── venv/  ← (No se sube al repo)
│
└── Proy25/
    ├── Informe_Proy_25.md
    ├── img/
    └── code/
        ├── requirements.txt
        └── venv/  ← (No se sube al repo)
```

---

## 🚀 Cómo configurar tu entorno local

### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/Expo.git
cd expo-262/"Proy 3"/code
```
*(o cambiá `Proy 3` por `Proy 13` o `Proy 25` según el proyecto que quieras ejecutar)*

---

### 2️⃣ Crear el entorno virtual
```bash
python -m venv venv
```

---

### 3️⃣ Activar el entorno virtual

#### En Windows (cmd o PowerShell)
```bash
venv\Scripts\activate
```

#### En Git Bash
```bash
source venv/Scripts/activate
```

#### En Linux/Mac
```bash
source venv/bin/activate
```

---

### 4️⃣ Instalar las dependencias
```bash
pip install -r requirements.txt
```

---

## 🧠 Uso con Jupyter Notebook o VS Code

Si trabajás con notebooks (`.ipynb`) o desde VS Code:

1. Abrí VS Code dentro de la carpeta del proyecto (por ejemplo `Proy3/code`).
2. Presioná `Ctrl + Shift + P` → escribí **“Select Interpreter”**.
3. Seleccioná el Python dentro de tu entorno virtual:
   ```
   .../Proy3/code/venv/Scripts/python.exe
   ```
4. Si usás Jupyter Notebook, registrá el kernel:
   ```bash
   python -m ipykernel install --user --name=venv-Proy3 --display-name "Python (Proy3)"
   ```
   *(Cambiá el nombre según el proyecto)*

---

## 💾 Actualizar dependencias
Si instalás nuevas librerías:
```bash
pip freeze > requirements.txt
```

---

## 🧹 Reglas generales
- No subas la carpeta `venv/` al repositorio.
- El `.gitignore` en la raíz de `Expo/` ya ignora todos los entornos virtuales.
- Cada proyecto debe mantener su propio `requirements.txt`.

---

**Autor:** Maximiliano Gómez Mallo  
**Repositorio base:** [GitHub - TU_USUARIO/Expo](https://github.com/TU_USUARIO/Expo)
