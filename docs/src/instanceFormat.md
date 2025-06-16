For the moment the function that reads the instance from file only support one data format for each problem type.

# Multi Commodity Network Design

For this problem the format is ... the one presented in [^1]
number_of_nodes number_of_arcs number_of_commodities
for each arc):
- head tail activation_cost capacity number_of_commodities
- for each commodity:( 
* commodty_index routing_cost capacity_for_commodity
)
for each commodity:(
- for each node:(
* node_index rhs_flow_constraint
)
)

# Generalized Assignment

The format for the Generalized Assignement Problem is the same of the one of OR-lib [^2].

 number of different problem sets (P)

 for each problem set p (p=1,...,P) in turn:
-	 number of agents (m), number of jobs (n)

 for each agent i (i=1,...,m) in turn:
  -   cost of allocating job j to agent i (j=1,...,n)

 for each agent i (i=1,...,m) in turn:
 -    resource consumed in allocating job j to agent i (j=1,...,n)

 resource capacity of agent i (i=1,...,m)

# Capacitated Warehose Location

The format for the Capacitated Warehouse Problem is the same of the one of OR-lib [^2]:

number of potential warehouse locations (m), number of customers (n)

for each potential warehouse location i (i=1,...,m): 
-	capacity, fixed cost

for each customer j (j=1,...,n): 
-	demand, cost of allocating

all of the demand of j to warehouse i (i=1,...,m)


# Unit Commitment

Also for the Unit Commitment Problem the data format is the same of OR-lib [^2].
First used and presented in [^3] and [^4]



[^1]: Crainic, T. G., Frangioni, A., and Gendron, B. Bundle-based relaxation methods for multicommodity capacitated fixed charge network design. Discrete Applied Mathematics, 112(1-3):pp. 73–99, 2001.
[^2]: http://people.brunel.ac.uk/~mastjjb/jeb/info.html
[^3]: Frangioni, C. Gentile, F. Lacalandra 'Solving Unit Commitment Problems with General Ramp Contraints' International Journal of Electrical Power and Energy Systems, to appear, 2008
[^4]: Frangioni, C. Gentile 'Solving Nonlinear Single-Unit Commitment Problems with Ramping Constraints' Operations Research 54(4), p. 767 - 775, 2006 These papers also describe in details the Unit Commitment model.
