# Nombre del ejecutable final
TARGET = run_flow2.out

# Librería estática
LIB = lib_flow.a

# Archivos fuente para la librería
LIB_SRC = alpha_3D.cu AXPBY_3D.cu beta_3D.cu CCMG_V_cycle.cu CG.cu \
          GSRB_Smooth_up_residual_3D.cu random_field_generation.cu \
          RHS_head_3D.cu stencil_head_3D.cu transf_operator_3D.cu \
          up_residual_3D.cu alphaU_3D.cu betaU_3D.cu omega_r_3D.cu omega_x_3D.cu

# Archivos objeto de la librería
LIB_OBJ = $(LIB_SRC:.cu=.o)

# Archivos fuente principales
MAIN_SRC = main_transport_JSON_input.cu compute_velocity_from_head_for_par2.cu

# Archivos objeto principales
MAIN_OBJ = $(MAIN_SRC:.cu=.o)

# Compilador
NVCC = nvcc

# Flags de compilación
NVCC_FLAGS = -std=c++11 -Iinclude --extended-lambda

# Librerías
LIBS = -lcublas -L. ./$(LIB)

# Habilitar compilación paralela con todos los núcleos disponibles
MAKEFLAGS += -j$(shell nproc)

# -------------------
# Reglas principales
# -------------------

all: $(TARGET)

# 1. Compilar ejecutable final
$(TARGET): $(MAIN_OBJ) $(LIB)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# 2. Generar la librería estática
$(LIB): $(LIB_OBJ)
	ar -rcs $@ $^

# 3. Compilar objetos de la librería
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -Xcompiler -static -c $< -o $@

# Limpiar archivos generados
clean:
	rm -f $(LIB_OBJ) $(MAIN_OBJ) $(TARGET) $(LIB)

# Recompilar desde cero
rebuild: clean all
