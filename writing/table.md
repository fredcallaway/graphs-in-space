---
title:  Foo
author: Fred Callaway
date: \today
style: Article
smallcaps:
    - A
    - B
    - ONE
---

\newcommand{\id}{\text{id}}
\newcommand{\sim}{\text{sim}}
\newcommand{\bump}{\text{bump}}
\newcommand{\weight}{\text{weight}}

\newcommand{\edge}[3][]{#2 \xrightarrow{#1} #3}



: Definition of symbols

|      Symbol      |                 Meaning                 |
|------------------+-----------------------------------------|
| $a, b, x, y, n$  | A node                                  |
| $ab$             | A node composed of $a$ and $b$          |
| $e$              | An edge label                           |
| $\edge[e]{x}{y}$ | The edge from $x$ to $y$ with label $e$ |
| $\id_x$          | The index vector of node $x$            |
| $\row_x$         | The row vector of node $x$              |


: Basic VectorGraph operations

+--------------------+-----------------------------------------+-------------------------------+
|   __Operation__    |               __Meaning__               |       __Implementation__      |
+====================+=========================================+===============================+
| $\bump(x, y, e)$   | increase the weight of $\edge[e]{x}{y}$ | $row_x +\!\!= \Pi_e \id_y$    |
+--------------------+-----------------------------------------+-------------------------------+
| $\weight(x, y, e)$ | the weight of $\edge[e]{x}{y}$          | $\cos(\row_x, \Pi_e (\id_y))$ |
+--------------------+-----------------------------------------+-------------------------------+
| $\sim(x, y)$       | similarity between $x$ and $y$          | $\cos(\row_x, \row_y)$        |
+--------------------+-----------------------------------------+-------------------------------+
