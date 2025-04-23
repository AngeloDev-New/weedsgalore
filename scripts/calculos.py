import numpy as np

class Globo:
    def __init__(self, *localizacoes : np.array) -> None:
        mDis, mPar = self.maior_distancia(*localizacoes)
        self.diametroGlobo = mDis
        self.bordas = np.array(mPar)
        self.excludes = []

    def centro(self)->np.array:
        return sum(self.bordas)/2
    
    def insideGlob(self, dist:np.array)-> bool:
        distance, _ = self.maior_distancia(dist, *self.bordas)
        return distance <= self.diametroGlobo
    
    def insideDistance(self,ponto:np.array,distanci_permitida:int)->bool:
        distancia_ponto_centro = self.distancia_euclidiana(self.centro(),ponto)
        return distanci_permitida >= distancia_ponto_centro
    

    @staticmethod
    def distancia_euclidiana(*dimensoes:np.array)->float:
        return np.sqrt(sum(np.square(dim[0] - dim[1]) for dim in dimensoes))
    
    def maior_distancia(self, *localizacoes:np.array)->tuple[float,tuple]:
        distancia_maior = 0
        par_maior = None
    
        for i, localA in enumerate(localizacoes):
            for j, localB in enumerate(localizacoes):
                if i < j:  # Evita calcular duas vezes a mesma distância
                    distancia = self.distancia_euclidiana(*zip(localA, localB))
                    if distancia > distancia_maior:
                        distancia_maior = distancia
                        par_maior = (localA, localB)
    
        return distancia_maior, par_maior  # Retorna a maior distância e o par correspondente


if __name__ == '__main__':
    # Pontos iniciais para definir o Globo
    a = (0, 0, 0)
    b = (10, 10, 10)
    c = (5, 5, 5)
    globo = Globo(a, b, c)
    
    # Criar um segundo globo para teste de exclusão
    d = (2, 2, 2)
    e = (4, 4, 4)


    
    # Teste de pontos dentro e fora dos Globos
    dentro = (10, 10, 10)  # Ponto dentro do Globo
    fora = (15, 15, 15)  # Ponto fora do Globo
    dentro_sub = (3, 3, 3)  # Ponto dentro do sub_globo
    
    print("Maior distância e bordas do Globo:", globo.diametroGlobo, globo.bordas)
    print('Ponto central do globo:',globo.centro())

    print("O ponto", dentro, "está dentro do Globo?-5", globo.insideDistance(dentro, 10))
    print("O ponto", fora, "está dentro do Globo?", globo.insideGlob(fora))

