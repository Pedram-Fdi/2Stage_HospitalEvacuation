class Scenario:
    def __init__(self, 
                 casualtyDemand, 
                 hospitalDisruption, 
                 patientDemand, 
                 patientDischargedPercentage, 
                 **attributes):
        
        self.CasualtyDemand = casualtyDemand  # Assign the specific casualtyDemand matrix for the scenario
        self.HospitalDisruption = hospitalDisruption  # Assign the specific hospitalDisruption matrix for the scenario
        self.PatientDemand = patientDemand  # Assign the specific patientDemand matrix for the scenario
        self.PatientDischargedPercentage = patientDischargedPercentage  # Assign the specific patientDischargedPercentage matrix for the scenario
        
        # Dynamically copy all other attributes
        for key, value in attributes.items():
            setattr(self, key, value)  # Set each attribute dynamically
