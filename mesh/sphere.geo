///////////////////////////////////////////////////////////////////
// Gmsh file for creating a finite element mesh
// In this case, we consider a sphere of radius R in a channel
// For a great tutorial on using Gmsh, see
// https://www.youtube.com/watch?v=aFc6Wpm69xo
///////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////

// element size at the solid interface
esc = 1e-2;

// element size for the solid
esa = 2e-2;

// radius of circle
R = 1;

////////////////////////////////////////////////////////////

// Create all of the points

// Points for the circle
Point(1) = {-R, 0, 0, esc};
Point(2) = {0, 0, 0, esa};
Point(3) = {R, 0, 0, esc};

// Create circle and lines
Circle(1) = {3, 2, 1};

Line(7) = {1, 2};
Line(8) = {2, 3};

Curve Loop(2) = {7, 8, 1};
Plane Surface(2) = {2};

// create physical lines (for Fenics)

// circle
Physical Curve(1) = {1};

// axis for solid
Physical Curve(6) = {7, 8};

// bulk (solid)
Physical Surface(11) = {2};
