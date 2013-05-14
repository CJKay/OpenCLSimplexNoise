#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__constant const char grad3[12][3] = {
	{ 1, 1, 0 }, { -1, 1, 0 }, { 1, -1, 0 }, { -1, -1, 0 },
	{ 1, 0, 1 }, { -1, 0, 1 }, { 1, 0, -1 }, { -1, 0, -1 },
	{ 0, 1, 1 }, { 0, -1, 1 }, { 0, 1, -1 }, { 0, -1, -1 }
};

double simplexdot2d(__constant const char *g, double x, double y) {
	return g[0] * x + g[1] * y;
}

double simplexdot3d(__constant const char *g, double x, double y, double z) {
	return g[0] * x + g[1] * y + g[2] * z;
}

double get2DRandom(double2 point, __constant const uchar *perm) {
	double n0, n1, n2;
	double F2 = 0.366025404;
	double s = (point.x + point.y) * F2;
	ulong i = (ulong)(point.x + s);
	ulong j = (ulong)(point.y + s);

	double G2 = 0.211324865;
	double t = (i + j) * G2;
	double X0 = i - t;
	double Y0 = j - t;
	double x0 = point.x - X0;
	double y0 = point.y - Y0;

	ulong i1, j1;
	if(x0 > y0) {
		i1 = 1;
		j1 = 0;
	}  else {
		i1 = 0;
		j1 = 1;
	}

	double x1 = x0 - i1 + G2;
	double y1 = y0 - j1 + G2;
	double x2 = x0 - 1.0 + 2.0 * G2;
	double y2 = y0 - 1.0 + 2.0 * G2;

	uchar ii = i & 255;
	uchar jj = j & 255;
	ulong gi0 = perm[ii + perm[jj]] % 12;
	ulong gi1 = perm[ii + i1 + perm[jj + j1]] % 12;
	ulong gi2 = perm[ii + 1 + perm[jj + 1]] % 12;

	double t0 = 0.5 - x0 * x0 - y0 * y0;
	
	if(t0 < 0 ) {
		n0 = 0.0;
	} else {
		t0 *= t0;
		n0 = t0 * t0 * simplexdot2d(grad3[gi0], x0, y0);
	}

	double t1 = 0.5 - x1 * x1 - y1 * y1;

	if(t1 < 0) {
		n1 = 0.0;
	} else {
		t1 *= t1;
		n1 = t1 * t1 * simplexdot2d(grad3[gi1], x1, y1);
	}

	double t2 = 0.5 - x2 * x2 - y2 * y2;

	if(t2 < 0) {
		n2 = 0.0;
	} else {
		t2 *= t2;
		n2 = t2 * t2 * simplexdot2d(grad3[gi2], x2, y2);
	}

	return 70.0 * (n0 + n1 + n2);
}

double get3DRandom(double3 point, __constant const uchar *perm) {
	double n0, n1, n2, n3;

	double F3 = 1.0 / 3.0;
	double s = (point.x + point.y + point.z) * F3;

	ulong i = (ulong)(point.x + s);
	ulong j = (ulong)(point.y + s);
	ulong k = (ulong)(point.z + s);

	double G3 = 1.0 / 6.0;
	double t = (i + j + k) * G3;
	double X0 = i - t;
	double Y0 = j - t;
	double Z0 = k - t;
	double x0 = point.x - X0;
	double y0 = point.y - Y0;
	double z0 = point.z - Z0;

	ulong i1, j1, k1;
	ulong i2, j2, k2;
	if(x0 >= y0) {
		if(y0 >= z0) {
			i1 = 1;
			j1 = 0;
			k1 = 0;
			i2 = 1;
			j2 = 1;
			k2 = 0;
		} else if(x0 >= z0) {
			i1 = 1;
			j1 = 0;
			k1 = 0;
			i2 = 1;
			j2 = 0;
			k2 = 1;
		} else {
			i1 = 0;
			j1 = 0;
			k1 = 1;
			i2 = 1;
			j2 = 0;
			k2 = 1;
		}
	} else {
		if(y0 < z0) {
			i1 = 0;
			j1 = 0;
			k1 = 1;
			i2 = 0;
			j2 = 1;
			k2 = 1;
		} else if(x0 < z0) {
			i1 = 0;
			j1 = 1;
			k1 = 0;
			i2 = 0;
			j2 = 1;
			k2 = 1;
		} else {
			i1 = 0;
			j1 = 1;
			k1 = 0;
			i2 = 1;
			j2 = 1;
			k2 = 0;
		}
	}
	
	double x1 = x0 - i1 + G3;
	double y1 = y0 - j1 + G3;
	double z1 = z0 - k1 + G3;
	double x2 = x0 - i2 + 2.0 * G3;
	double y2 = y0 - j2 + 2.0 * G3;
	double z2 = z0 - k2 + 2.0 * G3;
	double x3 = x0 - 1.0 + 3.0 * G3;
	double y3 = y0 - 1.0 + 3.0 * G3;
	double z3 = z0 - 1.0 + 3.0 * G3;
	
	uchar ii = i & 255;
	uchar jj = j & 255;
	uchar kk = k & 255;
	ulong gi0 = perm[ii + perm[jj + perm[kk]]] % 12;
	ulong gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12;
	ulong gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12;
	ulong gi3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12;
	
	double t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;

	if(t0 < 0) {
		n0 = 0.0;
	} else {
		t0 *= t0;
		n0 = t0 * t0 * simplexdot3d(grad3[gi0], x0, y0, z0);
	}

	double t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;

	if(t1 < 0) {
		n1 = 0.0;
	} else {
		t1 *= t1;
		n1 = t1 * t1 * simplexdot3d(grad3[gi1], x1, y1, z1);
	}

	double t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;

	if(t2 < 0) {
		n2 = 0.0;
	} else {
		t2 *= t2;
		n2 = t2 * t2 * simplexdot3d(grad3[gi2], x2, y2, z2);
	}

	double t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;

	if(t3 < 0) {
		n3 = 0.0;
	} else {
		t3 *= t3;
		n3 = t3 * t3 * simplexdot3d(grad3[gi3], x3, y3, z3);
	}
	
	return 32.0 * (n0 + n1 + n2 + n3);
}

double get2DValue(double2 point, double persistence, uchar octaves, double frequency, __constant const uchar *perm) {
	double curPersistence = persistence;
	double value = 0.0, scale = 0.0;
	double curFrequency, curAmplitude;
	
	for(uchar curOctave = 0; curOctave < octaves; ++curOctave) {
		curFrequency = pow(2.0, curOctave) * frequency;
		curAmplitude = pow(persistence, curOctave);

		value += get2DRandom(point * curFrequency, perm) * curAmplitude;
		scale += curAmplitude;
	}
	
	return value / scale;
}

double get3DValue(double3 point, double persistence, uchar octaves, double frequency, __constant const uchar *perm) {
	double curPersistence = persistence;
	double value = 0.0, scale = 0.0;
	double curFrequency, curAmplitude;
	
	for(uchar curOctave = 0; curOctave < octaves; ++curOctave) {
		curFrequency = pow(2.0, curOctave) * frequency;
		curAmplitude = pow(persistence, curOctave);

		value += get3DRandom(point * curFrequency, perm) * curAmplitude;
		scale += curAmplitude;
	}
	
	return value / scale;
}

__kernel void get2DMap(
	double2 delta,
	ulong2 dimensions,
	double persistence,
	uchar octaves,
	double frequency,
	__global double *result,
	__constant const uchar *perm
) {
	uint2 position;
	position.x = get_global_id(0);
	position.y = get_global_id(1);
	
	double2 point;
	point = convert_double2(position) * delta;
	
	result[position.y * dimensions.x + position.x] = get2DValue(point, persistence, octaves, frequency, perm);
}

__kernel void get3DMap(
	double3 delta,
	ulong3 dimensions,
	double persistence,
	uchar octaves,
	double frequency,
	__global double *result,
	__constant const uchar *perm
) {
	uint3 position;
	position.x = get_global_id(0);
	position.y = get_global_id(1);
	position.z = get_global_id(2);
	
	double3 point;
	point = convert_double3(position) * delta;
	
	result[position.z * dimensions.x * dimensions.y + position.y * dimensions.x + position.x] = get3DValue(point, persistence, octaves, frequency, perm);
}
