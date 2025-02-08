
https://github.com/user-attachments/assets/db29d4ce-3422-4dfd-b734-16f5bb0fb528
# BallHittingTargets
# Projectile Launcher Simulation  

## 1. Problem Formulation  

### 1.1 Physical Problem Statement  
The project simulates a projectile launcher system that must hit multiple circular targets in a 2D space while accounting for gravity and air resistance. The system must:  
- Detect circular targets from an input image  
- Calculate trajectories that successfully hit each target  
- Visualize both successful and failed attempts  

### 1.2 Governing Equations  
The motion is governed by a system of four first-order ODEs:  

#### For horizontal motion:  
- dx/dt = vx
- dvx/dt = -(b/m)vx 

#### For vertical motion:  
- dy/dt = vy
- dvy/dt = -g - (b/m)vy

Where:  
- x, y: position coordinates (m)
- vx, vy: velocity components (m/s)
- g: gravitational acceleration (9.81 m/s²)
- b: drag coefficient (0.5 kg/s)
- m: mass of projectile (1 kg) 

### 1.3 Initial Conditions  
- x(0) = 0
- y(0) = initial_height (20m)
- vx(0) = initial_vx (to be determined)
- vy(0) = initial_vy (to be determined) 

### 1.4 Boundary Conditions  
- Ground collision: y ≥ radius
- Target collision: √((x - xt)² + (y - yt)²) ≤ (r + rt)
Where (xt, yt) is target position and rt is target radius
---

## 2. Method of Solution  

### 2.1 Circle Detection Algorithm  

#### 1. Image Preprocessing:  
- Convert to grayscale  
- Apply Gaussian blur for noise reduction  

#### 2. Edge Detection:  
- Apply Sobel operator for gradient computation  
- Perform non-maximum suppression  
- Apply threshold to obtain binary edge image  

#### 3. Circle Detection (Modified Hough Transform):  
For each edge pixel (x, y):  
- Use gradient direction to predict circle centers  
- Accumulate votes in parameter space (a, b, r)  
- Find local maxima in accumulator space  
- Apply threshold and non-maximum suppression  
- Scale detected circles to simulation space  

### 2.2 Trajectory Computation  

#### 2.2.1 RK4 Integration  
The 4th order Runge-Kutta method is implemented with a fixed time step:  

For state vector  Y = [x, v_x, y, v_y] :  

- k1 = f(t, Y)
- k2 = f(t + dt/2, Y + dt*k1/2)
- k3 = f(t + dt/2, Y + dt*k2/2)
- k4 = f(t + dt, Y + dt*k3)
- Y(t + dt) = Y(t) + (dt/6)(k1 + 2k2 + 2k3 + k4) 

#### 2.2.2 Shooting Method Implementation  
1. For each target:  
   - Generate random initial velocities (v_x, v_y)  
   - Compute trajectory using RK4  
   - Check for target collision  
   - Repeat until success or max attempts reached  

---

## 3. Algorithm Properties  

### 3.1 RK4 Method Properties  
- Local truncation error: \( O(h^5) \)  
- Global truncation error: \( O(h^4) \)  
- Stable for small time steps  
- Self-starting method  
- Computationally more expensive than lower-order methods  

### 3.2 Circle Detection Properties  
- Rotation invariant  
- Scale dependent (min/max radius parameters)  
- Robust to partial occlusion  
- Sensitive to threshold parameters  
- Computational complexity: O(n*m*r) where n, m are image dimensions and r is the radius range  

### 3.3 Shooting Method Properties  
- Monte Carlo approach  
- No guarantee of convergence  
- Success probability depends on:  
  - Target size and position  
  - Initial velocity ranges  
  - Number of attempts allowed  

---

## 4. Test Cases  
test cases are given in TestCases.py 

---

## 5. Results Interpretation  

### 5.1 Known Limitations  

#### 1. Physics Model:  
- Simplified air resistance model  
- No wind effects  
- No spin dynamics  

#### 2. Circle Detection:  
- Sensitive to image quality  
- May detect false positives  
- Limited to circular targets  

#### 3. Shooting Method:  
- Random search is inefficient  
- No optimization of initial velocities  
- May fail for difficult target positions  
