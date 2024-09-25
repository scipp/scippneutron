from abc import ABCMeta, abstractmethod
from typing import Any

import scipp as sc


class SampleShape(metaclass=ABCMeta):
    @abstractmethod
    def beam_intersection(
        self, start_point: sc.Variable, direction: sc.Variable
    ) -> sc.Variable:
        '''Computes the length of the intersection between the shape and the beam
        starting at `start_point` and travelling in the direction `direction`.'''
        pass

    @property
    @abstractmethod
    def volume(self) -> sc.Variable:
        '''Volume of the shape'''
        pass

    @abstractmethod
    def quadrature(self, kind: Any) -> tuple[sc.Variable, sc.Variable]:
        '''Returns quadrature points and weights for evaluating integrals over
        the shape. The method returns a tuple where the first entry is
        an array containing vectors representing points in the shape and the
        second entry is an array containing the weights associated with
        the points.

        Parameters
        -----------
        kind:
            if the shape supports different kinds of quadratures
            this argument denotes which one to use
        '''
        pass
