//
// Created by brian on 11/20/18.
//

#include <cmath>
#include <iostream>
#include <cuda.h>

const float PI = 3.14159265358979f;

__device__ __host__ Complex::Complex() : real(0.0f), imag(0.0f) {}

__device__ __host__ Complex::Complex(float r) : real(r), imag(0.0f) {}

__device__ __host__ Complex::Complex(float r, float i) : real(r), imag(i) {}

__device__ __host__ Complex Complex::operator+(const Complex &b) const {
    return Complex(this->real+b.real, this->imag+b.imag);
}

__device__ __host__ Complex Complex::operator-(const Complex &b) const {
    return Complex(this->real-b.real, this->imag-b.imag);
}

__device__ __host__ Complex Complex::operator*(const Complex &b) const {
    return Complex((this->real*b.real)-(this->imag*b.imag), (this->real*b.imag)+(this->imag*b.real));

}

__device__ __host__ Complex Complex::mag() const {
    return Complex(sqrt((this->real*this->real)+(this->imag*this->imag)));
}

__device__ __host__ Complex Complex::angle() const {
    return Complex(sqrt(atan(this->imag/this->real)));
}

__device__ __host__ Complex Complex::conj() const {
    return Complex(this->real, -1*(this->imag));
}

std::ostream& operator<< (std::ostream& os, const Complex& rhs) {
    Complex c(rhs);
    if(fabsf(rhs.imag) < 1e-10) c.imag = 0.0f;
    if(fabsf(rhs.real) < 1e-10) c.real = 0.0f;

    if(c.imag == 0) {
        os << c.real;
    }
    else {
        os << "(" << c.real << "," << c.imag << ")";
    }
    return os;
}