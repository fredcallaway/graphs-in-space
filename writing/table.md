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
\newcommand{\cosine}{\text{cosine}}
\newcommand{\sim}{\text{sim}}
\newcommand{\bump}{\text{bump}}
\newcommand{\weight}{\text{weight}}



+--------------------+----------------------------------------------+-----------------------------+
|   __Operation__    |                 __Meaning__                  |      __Implementation__     |
+====================+==============================================+=============================+
| $\bump(x, y, e)$   | increase weight of edge $e$ from $x$ to $y$  | $row_x += \Pi_e id_y$       |
+--------------------+----------------------------------------------+-----------------------------+
| $\weight(x, y, e)$ | retrieves weight of edge $e$ from $x$ to $y$ | $cosine(row_x, \Pi_e id_y)$ |
+--------------------+----------------------------------------------+-----------------------------+
| $\sim(x, y)$       | similarity between $x$ and $y$               | $\cosine(row_x, row_y)$     |
+--------------------+----------------------------------------------+-----------------------------+

: Sample grid table.


