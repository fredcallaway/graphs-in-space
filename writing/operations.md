


|   __Operation__    |               __Meaning__               |        __Implementation__       |
|--------------------|-----------------------------------------|---------------------------------|
| $\bump(x, y, e)$   | increase the weight of $\edge[e]{x}{y}$ | $\row_x +\! = \, \Pi_e (\id_y)$ |
| $\weight(x, y, e)$ | the weight of $\edge[e]{x}{y}$          | $\cos(\row_x, \Pi_e (\id_y))$   |
| $\simil(x, y)$       | similarity between $x$ and $y$          | $\cos(\row_x, \row_y)$          |

: Basic VectorGraph operations
