from pyrokientics import pyro
def caluclate_kperp(pyro):
    pyro.load_gk_output()
    pyro.load_metric_terms(ntheta=pyro.numerics.ntheta)
    ky = pyro.numerics.ky
    nperiod = pyro.numerics.nperiod + 1
    theta0 = pyro.numerics.theta0

    kperp2 = pyro.metric_terms.k_perp(ky, theta0, nperiod)
    fields = pyro.gk_output.data["eigenfunctions"]
    field_squared = (np.abs(fields.isel(time=-1))**2)
    field_squared = np.squeeze(field_squared)
    jacobian = pyro.metric_terms.Jacobian
    intfield2 = (jacobian*field_squared).integrate(coord="theta")
    kperp2_field = (kperp**2*jacobian*field_squared).integrate(coord="theta")/intfield2      

    return k_perp_pyro


def calcualte_kacobian():



