import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

def Gaussian_Orthogonal_Ensemble(n, **kwargs):
    
    '''
    Generates a nxn Gaussian Orthogonal Ensemble
    '''
    
    # Define a normally distributed array
    H = np.random.normal(0, 1, size=(n,n))
    
    # Make it symmetric
    H = np.triu(H) + np.triu(H, k=1).T
    
    # Multiply the diagonal values by a factor of sqrt(2)
    np.fill_diagonal(H, np.sqrt(2) * np.diag(H))
    
    return H

def eigenvalues(n, **kwargs):
    
    '''
    Computes the eigenvalues for a nxn Gaussian Orthogonal Ensemble
    '''
    
    eigs = np.linalg.eig(Gaussian_Orthogonal_Ensemble(n))[0]
    
    return eigs

def sqrt_interp(t, p1, p2, **kwargs):

    """
    Square root interpolate two points
    
    Inputs
        t : location between p1[0] and p2[0]
        p1 : point 1
        p2 : point 2

    Returns
        sqrt_interp : sqrt interpolated value at location t
    """
    
    # Create a copy of each point
    pt1 = p1.copy()
    pt2 = p2.copy()
    
    # Project point 1 to x axis
    pt1[1] = 0
    pt2[1] = p2[1] - p1[1]
    
    # Calculate positive and reflect over y axis if needed
    if pt2[1] - pt1[1] < 0:
        
        sqrt_interp = -1 * np.sqrt(np.interp(t, [pt1[0], pt2[0]], [pt1[1]**2, pt2[1]**2]))
    
    else:
        
        sqrt_interp = np.sqrt(np.interp(t, [pt1[0], pt2[0]], [pt1[1]**2, pt2[1]**2]))
    
    # Undo projection
    sqrt_interp += p1[1]
    
    # Reflect and translate
    sqrt_interp = -1 * sqrt_interp + p1[1] + p2[1]
    
    return sqrt_interp

def n_Brownian_Motions_GOE(n, n_i, T, steps, dt):
    
    '''
    Simulates the Brownian Motion of n particles with starting positions of eigenvalues of a GOE.

    Inputs
        n: number of particles
        n_i : initial conditions of particles
        T : Total time
        steps : number of square root interpolations of each driver
        dt : time between random motions

    Outputs
        positions : Raw position data of the particle motion
        positions_interp: Sliced positions to sqrt interpolate
    '''
    
    # Create an array of positions +  assign initial values
    positions = np.zeros((int(np.ceil((T+dt) / dt)), n))
    positions[0] = n_i
    
    # For each timestep after the initial
    for i in range(len(positions) - 1):
        
        # Generate some eigenvalues for the motion
        eigs = np.sort(eigenvalues(n))
        
        # Calculate the Coulombic interation
        for j in range(n):
            
            coulombic = 0
            
            for k in range(n):
            
                if k != j:
                
                    coulombic += 1 / (positions[i, j] - positions[i, k])
            
            # Assign that interaction to each particle and calculate the path for that timestep
            positions[i+1, j] = positions[i, j] + np.sqrt(2 * dt / n) * eigs[j] + coulombic / n
    
    # Slice out the wanted values
    positions_interp = positions[::int(T / dt) // steps]
        
    return positions, positions_interp

def complex_function(t,y):

    """
    Derivative definition for the ODE solver

    Inputs
        t : time (float)
        y : solution, 1D (ndarray)

    Outputs
        dy : derivative expression, 1D (ndarray)
    """
    
    x = y[0]
    y = y[1]
    
    dxdt = 0
    dydt = 0
    
    for i in range(n):
        
        f = np.interp(t, t_brownian, interp_brownian[i])

        dxdt += -2 * (x - f) / ( (x - f) ** 2 + y ** 2 )
    
        dydt += 2 * y / ( (x - f) ** 2 + y ** 2 )
    
    dxdt /= n
    dydt /= n

    return np.array([dxdt, dydt])

############################
# Brownian Motion Conditions
# Only change constants in this section
############################
n = 3                                        # Number of curves
n_i = np.sort(eigenvalues(n))
T = 1                                        # Total runtime
steps = 10                                   # Stepsize for sqrt interpolation
dt = 0.01                                    # IMPORTANT: dt * steps * integer == T
interp_mesh_size = 1000                      # Keep this ~1000 or higher
anim_times = np.linspace(0,T, 101)           # Make this higher to adjust the SLE curve(s) resolution
x_grid_density = 0.5                         # Approximate uniform spacing of x grid values
y_points = np.logspace(-31, np.log10(2), 31) # logarithmically space #3 points between 10^(#1) and 10^(#2)
starting_singularity_height = 1e-3           # y-value to start the SLE curves at

#############################################
# Generate + Interpolate Brownian Motion
#############################################

# Complete Brownian motion
positions, positions_interp = n_Brownian_Motions_GOE(n, n_i, T, steps, dt)

# Create blank arrays to store interpolated values and time expression
interp_brownian = [[] for i in range(n)]
t_brownian = np.array([])

# Overly complicated way of sqrt interpolation      
for i in range(n):
    
    brownian_pos = np.array([])

    for j in range((int(T / dt) // steps)):
        
        # Calculate where each 'point' is [t, y], same as in the example above.
        point1 = [T - (j + 1) * dt * steps, positions[-1 * (j+1) * steps-1, i]]
        point2 = [T - j * dt * steps, positions[-1 * j * steps-1, i]]
        
        if i == 0:
            t_brownian = np.append(t_brownian, np.linspace(point1[0], point2[0], interp_mesh_size))
            
        brownian_pos = np.concatenate((brownian_pos, sqrt_interp(np.linspace(point1[0], point2[0], interp_mesh_size), point1, point2)))  
        
    interp_brownian[i].append(brownian_pos)      

# Resizing arrays because I don't want to fix the code
interp_brownian = np.array(interp_brownian)
interp_brownian = np.squeeze(interp_brownian, axis=1)

# Time reverse   
t_brownian = np.flip(t_brownian)

###################
# Simulate the Grid
###################

# The x points are spaced depending on how the eigen values
eigenvalue_range = n_i[-1] - n_i[0]
x_points = np.linspace(n_i[0] - 0.1 * eigenvalue_range,n_i[-1] + 0.1 * eigenvalue_range, int(1.2 * eigenvalue_range / x_grid_density))

# Create a 4D array to store solutions over time
sol_4D = np.zeros((len(anim_times), len(x_points), len(y_points), 2))

# Assign initial conditions in the slowest way possible 
for i in range(len(x_points)):
    
    for j in range(len(y_points)):
        
        sol_4D[0,i,j] = np.array([x_points[i], y_points[j]])
        
# Grid simulating
for k in range(len(anim_times) - 1):
    
    print(f'{k+1}/{len(anim_times) - 1}')

    for i in range(len(x_points)):
    
        for j in range(len(y_points)):
            
            t_span = [k * T / len(anim_times), (k + 1) * T / len(anim_times)]
            
            z0 = np.array([sol_4D[k,i,j,0],sol_4D[k,i,j,1]])
            
            sol = solve_ivp(complex_function, t_span, z0, method='BDF', atol=1e-15, rtol=1e-12)
            
            x = sol.y[0]
            y = sol.y[1]
        
            sol_4D[k+1,i,j,0] = x[-1]
            sol_4D[k+1,i,j,1] = y[-1]

#######################
# Track the singularity
#######################

# singularity_sol = np.zeros((n, len(anim_times), 2))

# for i in range(n):

#         singularity_sol[i, 0] = np.array([interp_brownian[i, -1],1e-6])

# # Singularity simulating
# for i in range(n):

#     for k in range(len(anim_times) - 1):
    
#         # print(f'{k+1}/{len(anim_times) - 1}')
            
#         t_span = [k * T / len(anim_times), (k + 1) * T / len(anim_times)]
            
#         z0 = np.array([singularity_sol[i,k,0],singularity_sol[i,k,1]])
            
#         sol = solve_ivp(complex_function, t_span, z0, method='BDF', atol=1e-15, rtol=1e-12)
            
#         x = sol.y[0]
#         y = sol.y[1]
        
#         singularity_sol[i,k+1,0] = x[-1]
#         singularity_sol[i,k+1,1] = y[-1]

SLE_paths = [[] for i in range(n)]
for i in range(n):

    # Starting point
    single_path = [[[n_i[i], starting_singularity_height]]]

    for k in range(len(anim_times)):

        t_span = [k * T / len(anim_times), (k + 1) * T / len(anim_times)]
        
        single_path.append([[np.interp(t_span[-1], t_brownian, interp_brownian[i]), starting_singularity_height]])

        for j in range(len(single_path[k])):

            z0 = np.array(single_path[k][j])
            
            sol = solve_ivp(complex_function, t_span, z0, method='BDF', atol=1e-15, rtol=1e-12)

            x = sol.y[0]
            y = sol.y[1]

            single_path[k+1].append([x[-1],y[-1]])
    
    SLE_paths[i] = single_path

print(len(SLE_paths[0]))
print(len(SLE_paths))
print(SLE_paths[0])
print(SLE_paths[0][1][0][0])
print(np.array(SLE_paths[0][3][:])[:,0])

###########
# Animating
###########

def generate_plot(frame):
    print(frame)
    """
    Animation function that draws a frame for the gif. 

    Inputs
        frame : the number of the frame in the animation. Used for iterating through arrays

    Returns
        plt : frame added to animation
    """

    # Clear the current axes
    plt.cla()
    
    # Plot the grid
    for i in range(len(x_points)):
    
        plt.plot(sol_4D[frame,i,:,0], sol_4D[frame,i,:,1], lw=0.4, color='tab:blue')
        
    for j in range(len(y_points)):
            
        plt.plot(sol_4D[frame,:,j,0], sol_4D[frame,:,j,1], lw=0.4, color='tab:blue')    
    
    for i in range(n):
    
        # Plot the singularity path
        plt.plot(np.array(SLE_paths[i][frame][:])[:,0], np.array(SLE_paths[i][frame][:])[:,1], color='tab:red')
        
            
    # Set plot title and labels, if desired
    plt.title(f"t={anim_times[frame]:0.2f}")

    # Return the plot or figure object
    return plt

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10,10))

# Create the animation
anim = animation.FuncAnimation(fig, generate_plot, frames=len(anim_times), interval=200)

# Set up the writer
writer = PillowWriter(fps=10)

# Save the animation as an gif file
anim.save('final_animation_grid_n_brownian_driver.gif', writer=writer)
plt.close()