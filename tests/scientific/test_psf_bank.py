import numpy as np
import pytest
from scipy.special import ndtr

from spotsolve.inference import PixelPSFBank
from spotsolve.inference._numerics import ROIGrid


@pytest.fixture(scope='module')
def bank():
    z=np.linspace(0,.6,7);xy=np.arange(-22,22.01,.5)
    response=[]
    for depth in z:
        s=1.2+3*depth
        # Pixel-integrated Gaussian sample; floor only negligible distant tails.
        a=ndtr((xy+.5)/s)-ndtr((xy-.5)/s)
        response.append(np.maximum(a[:,None]*a[None,:],1e-100))
    return PixelPSFBank(z,xy,np.array(response))


def test_table_reproduces_samples_and_has_analytic_derivatives(bank):
    y,x=np.meshgrid(bank.offsets_px[40:48],bank.offsets_px[40:48],indexing='ij')
    actual=bank.evaluate_offsets(y,x,.3)[0]
    np.testing.assert_allclose(actual,bank.responses[3,40:48,40:48],rtol=2e-13)
    args=[np.array([.23,-1.28]),np.array([-.42,2.13]),.337]
    out=bank.evaluate_offsets(*args)
    for k in range(3):
        left=list(args);right=list(args)
        left[k]=args[k]-1e-6;right[k]=args[k]+1e-6
        numerical=(bank.evaluate_offsets(*right)[0]-bank.evaluate_offsets(*left)[0])/2e-6
        np.testing.assert_allclose(numerical,out[k+1],rtol=2e-6,atol=1e-9)
    with pytest.raises(ValueError,match='outside calibration'):
        bank.evaluate_offsets(23,0,.3)


def test_psf_preserves_tail_loss_when_roi_changes(bank):
    full=bank.evaluate(ROIGrid.from_shape((21,21)),10.2,9.7,.4)[0]
    cropped=bank.evaluate(ROIGrid.from_shape((13,13)),6.2,5.7,.4)[0]
    np.testing.assert_allclose(cropped,full[4:17,4:17],atol=1e-15,rtol=1e-13)
    assert cropped.sum()<full.sum()
