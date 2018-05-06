// https://medium.com/coding-with-clarity/speeding-up-python-and-numpy-c-ing-the-way-3b9658ed78f4
// https://github.com/mattfowler/PythonCExtensions

// #include <Python/Python.h>
#include <Python.h>
#include <vector>
#include <numeric>
#include <iterator>
#include <iostream>

#include <Python.h>

extern "C" {
    void initlaxhopf(void);
}

static std::vector<double> propagateHopf(std::vector<double> v,double a, int T)
{
    std::vector<double> u(v.size());
    // Vector pointers to avoid copying
    std::vector<double>* prev=&v;
    std::vector<double>* next=&u;
    std::vector<double>* temp=prev;

    // Iterate T times
    for(int t=0; t<T; t++){
      // For first element in vector
      // ....
      // For elements inside vector
      for(int i =1; i<u.size()-1; i++){
        next->at(i)= prev->at(i) - 0.25*a*(pow(prev->at(i+1),2) - pow(prev->at(i-1),2))
        + 0.125*pow(a,2)*(  (prev->at(i+1) + prev->at(i))*(pow(prev->at(i+1),2) - pow(prev->at(i),2)) - (prev->at(i) + prev->at(i-1))*(pow(prev->at(i),2) - pow(prev->at(i-1),2))  );
      }
      // For last element in vector
      // ...

      // Switch pointers, new status now is old status next
      temp=prev;
      prev=next;
      next=temp;
    }
    // Return next which is now prev...
    return *prev;
}

static PyObject * propagate_Hopf(PyObject *self, PyObject* args)
{
    // Load args
    PyObject* input;
    PyObject* python_T;
    PyObject* python_a;
    PyArg_ParseTuple(args, "OOO", &input, &python_a, &python_T);

    // Convert to C types
    int size = PyList_Size(input);
    std::vector<double> list(size);
    for(int i = 0; i < size; i++) {
        list[i] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(input, i));
    }
    double a = PyFloat_AS_DOUBLE(python_a);
    int T = PyInt_AS_LONG(python_T);

    // Propagate T times
    std::vector<double> data = propagateHopf(list, a, T);

    // Convert back to Py-Types
    // https://gist.github.com/rjzak/5681680
    PyObject* listObj = PyList_New( data.size() );
    for (unsigned int i = 0; i < data.size(); i++) {
      PyObject *num = PyFloat_FromDouble( (double) data[i]);
      if (!num) {
        Py_DECREF(listObj);
      }
      PyList_SET_ITEM(listObj, i, num);
    }

    return listObj;
}

static PyMethodDef laxhopf_methods[] = {
	{"lax_hopf", propagate_Hopf,	METH_VARARGS,
	 "Propagates vector using Lax W. method and assuming Hopfs equation."},
	{NULL,		NULL}		/* sentinel */
};

extern void initlaxhopf(void)
{
	PyImport_AddModule("laxhopf");
	Py_InitModule("laxhopf", laxhopf_methods);
}

int main(int argc, char **argv)
{
	Py_SetProgramName(argv[0]);

	Py_Initialize();

	initlaxhopf();

	Py_Exit(0);
}
