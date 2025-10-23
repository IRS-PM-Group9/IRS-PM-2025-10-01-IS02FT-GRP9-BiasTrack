# -------------------------------
# Utility Functions for Dashboards
# -------------------------------

class Utility:

    # Add Total Pay column to compute data statistics
    def preprocess_data(self, df):
        df['Total Pay'] = df['BasePay'] + df['Bonus']
        df.drop(columns=['BasePay', 'Bonus'], inplace=True)
        return df

    # Show the female male count within the organization
    def female_male_count(self, df):
            male_count=df[df['Gender']=='Male'].shape[0]
            female_count=df[df['Gender']=='Female'].shape[0]
            return male_count,female_count
        
    #Show the job roles
    def job_roles(self,df):
        sorted_job_roles=df['JobTitle'].drop_duplicates().sort_values()
        return sorted_job_roles
        
    # Show the avg salary for female male for the job roles
    def avg_pay_by_role(self,df):
        avg_pay_by_role=df.groupby(['JobTitle','Gender','Seniority'])['Total Pay'].mean().reset_index()
        return avg_pay_by_role

    # Pay Gap by role
    def pay_gap_calculator(self,df):
        avg_pay= self.avg_pay_by_role(df)

        # Group by JobTitle and Gender only, averaging over Seniority
        avg_pay_simple = avg_pay.groupby(['JobTitle', 'Gender'])['Total Pay'].mean().reset_index()

        pivot = avg_pay_simple.pivot(index='JobTitle', columns='Gender', values='Total Pay')
        pivot['Pay_Gap_%'] = ((pivot['Male'] - pivot['Female']) / pivot['Female']) * 100
        pivot=pivot.reset_index()
        return pivot
        
    # Overall Pay Gap of the organization
    def overall_pay_gap(self,df):
        avg_pay_female=df[df['Gender']=='Female']['Total Pay'].mean()
        avg_pay_male=df[df['Gender']=='Male']['Total Pay'].mean()
        overall_pay_gap_org=((avg_pay_male - avg_pay_female)/avg_pay_female)*100
        return overall_pay_gap_org
        
    # Summary Dashboard
    def summary_dashboard_metrics(self,df):
        male,female=self.female_male_count(df)
        overall_pay=self.overall_pay_gap(df)
        pay_gap_cal=self.pay_gap_calculator(df)

        return{
            'male_count':male,
            'female_count':female,
            'overall_pay_gap':overall_pay,
            'pay_gap_by_role':pay_gap_cal
        }