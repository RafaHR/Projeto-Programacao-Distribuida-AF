### Vetorização para criptografia de imagens sensíveis
---


## Objetivos

Este trabalho tem como foco principal implementar, com numpy, o processamento vetorizado em um sistema de criptografia para imagens sensíveis, de modo a, comparar a pipeline de processamento com e sem o sistema de vetorização


## Contextualização

Este projeto surgiu como um ponto de melhoria ao trabalho de conclusão de curso intitulado "Computação Quântica e Criptografia Pós-Quântica Aplicadas à Segurança de Dados Sensíveis".

Nesse contexto, o TCC propõe implementar um sistema de proteção de imagens sensíveis que combina três módulos principais.

- 1 - Geradores Quânticos de Números Aleatórios (QRNG)
- 2 - Sistema de cifragem baseados em operações de XOR e mapa caótico logístico.
- 3 - Encapsulamento de chaves com o algoritmo pós-quântico CRYSTAL-Kyber 512.

## Recursos utilizados

As bibliotecas utilizadas foram:

- Numpy
- OpenCV
- Matplotlib
- Time

Para rodar é necessário instalar as bibliotecas principais como Numpy, OpenCV e Matplotlib

A principio, esse projeto foi rodado criando um ambiente separado

```
conda create -n "vector" python=3.12
```

Criado o ambiente, é necessário ativa-lo e instalar as dependências

```
conda activate vector
```

E instalar as dependências através do comando

```
pip install -r requirements.txt
```

## Explicação

