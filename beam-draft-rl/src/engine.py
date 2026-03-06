import pint

ureg = pint.UnitRegistry()

class PhysicsEngine:
    def __init__(self):
        self.ureg = ureg

    def validate_equilibrium(self, length, forces, reactions):
        """
        Validates if the sum of forces and moments are zero.
        forces: list of dicts {'value': '10N', 'position': '2m'}
        reactions: list of dicts {'value': '5N', 'position': '0m'}
        length: '5m'
        """
        L = self.ureg(length)
        
        total_force = 0 * self.ureg.newton
        total_moment = 0 * self.ureg.newton * self.ureg.meter

        for f in forces:
            val = self.ureg(f['value'])
            pos = self.ureg(f['position'])
            total_force += val
            total_moment += val * pos

        for r in reactions:
            val = self.ureg(r['value'])
            pos = self.ureg(r['position'])
            total_force += val
            total_moment += val * pos

        force_is_zero = abs(total_force.to(self.ureg.newton).magnitude) < 1e-6
        moment_is_zero = abs(total_moment.to(self.ureg.newton * self.ureg.meter).magnitude) < 1e-6

        return force_is_zero and moment_is_zero

    def solve_simply_supported_beam(self, length, force_val, force_pos):
        """
        Calculates reactions for a simply supported beam with one point load.
        """
        L = self.ureg(length)
        P = self.ureg(force_val)
        a = self.ureg(force_pos)
        
        # Ry = P * (L - a) / L
        # R0 = P - Ry
        
        R_L = P * (L - a) / L
        R_R = P * a / L
        
        return {
            'R0': (-R_L).to(self.ureg.newton), # Reaction at 0
            'RL': (-R_R).to(self.ureg.newton)  # Reaction at L
        }
