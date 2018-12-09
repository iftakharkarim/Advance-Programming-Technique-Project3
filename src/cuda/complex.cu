//
// Created by brian on 11/20/18.
//

#include "complex.h"

#include <cmath>

const float PI = 3.14159265358979f;

Complex::Complex() : real(0.0f), imag(0.0f) {}

Complex::Complex(float r) : real(r), imag(0.0f) {}

Complex::Complex(float r, float i) : real(r), imag(i) {}

// simple addition
Complex Complex::operator+(const Complex &b) const {
    Complex result;
    result.real = this->real+b.real;
    result.imag = this->imag+b.imag;
    return result;
}

Complex Complex::operator-(const Complex &b) const {
    Complex result;
    result.real = this->real-b.real;
    result.imag = this->imag-b.imag;
    return result;
}

Complex Complex::operator*(const Complex &b) const {
    Complex result;
    result.real = (this->real*b.real)-(this->imag*b.imag);
    result.imag = (this->real*b.real)+(this->imag*b.imag);
    return result;
}

// this is always real, never imaginary though?
Complex Complex::mag() const {
    Complex result;
    result.real = sqrt((this->real*this->real)+(this->imag*this->imag));
    return result;
}

Complex Complex::angle() const {
    Complex result;
    result.real = atan(this->imag/this->real);
    return result;
}

Complex Complex::conj() const {
    Complex result;
    result.real = this->real;
    result.imag = -1*(this->imag);
    return result;
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