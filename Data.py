import pandas as pd
import numpy as np

def generate_institutional_yearly_trends():
    """
    Generate institutional trends from 2001-2025 that affect all students
    """
    years = list(range(2001, 2026))  # 2001-2025 inclusive
    
    np.random.seed(42)
    
    # GPA trend: Grade inflation over 25 years
    base_gpa_trend = np.linspace(0, 0.18, 25)
    gpa_fluctuations = np.random.normal(0, 0.05, 25)
    yearly_gpa_effects = base_gpa_trend + gpa_fluctuations
    
    # Major events
    yearly_gpa_effects[7:9] -= 0.08    # 2008-2009 financial crisis
    yearly_gpa_effects[11:14] += 0.06  # 2012-2014 policy improvements
    yearly_gpa_effects[19:21] -= 0.04  # 2020-2021 COVID
    yearly_gpa_effects[21:] += 0.03    # 2022+ recovery
    
    # Dropout rates (decreasing over time)
    base_dropout_trend = np.linspace(0.018, 0.008, 25)
    dropout_fluctuations = np.random.normal(0, 0.002, 25)
    yearly_dropout_rates = base_dropout_trend + dropout_fluctuations
    yearly_dropout_rates[7:9] += 0.005   # Crisis increases dropouts
    yearly_dropout_rates[19:21] += 0.003 # COVID increases dropouts
    yearly_dropout_rates = np.maximum(yearly_dropout_rates, 0.003)
    
    # Early graduation rates
    base_early_grad_trend = np.linspace(0.015, 0.045, 25)
    early_grad_fluctuations = np.random.normal(0, 0.005, 25)
    yearly_early_grad_rates = base_early_grad_trend + early_grad_fluctuations
    yearly_early_grad_rates[11:14] += 0.01  # Policy flexibility
    yearly_early_grad_rates[19:21] -= 0.005 # COVID reduces early grad
    yearly_early_grad_rates = np.clip(yearly_early_grad_rates, 0.01, 0.08)
    
    return dict(zip(years, zip(yearly_gpa_effects, yearly_dropout_rates, yearly_early_grad_rates)))

def generate_gender_and_effects(student_id):
    """Generate gender with subtle academic performance effects"""
    student_seed = hash(student_id + "_gender") % (2**32)
    np.random.seed(student_seed)
    
    # SE enrollment: slight male bias
    gender = np.random.choice(['Male', 'Female'], p=[0.58, 0.42])
    
    if gender == 'Female':
        gpa_modifier = np.random.normal(0.09, 0.06)
        early_grad_modifier = 0.88
        dropout_modifier = 0.82
        credit_load_modifier = 0.97
    else:
        gpa_modifier = np.random.normal(-0.06, 0.06)
        early_grad_modifier = 1.18
        dropout_modifier = 1.12
        credit_load_modifier = 1.04
    
    return gender, gpa_modifier, early_grad_modifier, dropout_modifier, credit_load_modifier

def generate_student_performance_profile(student_id, enrollment_year):
    """Generate student archetype and academic journey"""
    student_seed = hash(student_id + str(enrollment_year)) % (2**32)
    np.random.seed(student_seed)
    
    archetype_roll = np.random.random()
    
    if archetype_roll < 0.05:
        return generate_struggling_student()
    elif archetype_roll < 0.10:
        return generate_super_achiever()
    elif archetype_roll < 0.15:
        return generate_inconsistent_student()
    elif archetype_roll < 0.20:
        return generate_late_bloomer()
    elif archetype_roll < 0.25:
        return generate_early_struggler()
    elif archetype_roll < 0.30:
        return generate_burnout_student()
    elif archetype_roll < 0.35:
        return generate_part_time_student()
    elif archetype_roll < 0.40:
        return generate_overloader_student()
    elif archetype_roll < 0.45:
        return generate_repeat_courses_journey()
    elif archetype_roll < 0.50:
        return generate_gap_semester_journey()
    elif archetype_roll < 0.55:
        return generate_working_student_journey()
    elif archetype_roll < 0.60:
        return generate_transfer_journey()
    else:
        return generate_normal_student_with_noise()

def generate_struggling_student():
    semesters = []
    archetype = "struggling"
    credit_pattern = [8, 10, 9, 11, 10, 12, 11, 13]
    
    for sem in range(8):
        attempted = credit_pattern[sem] + np.random.randint(-2, 3)
        attempted = max(attempted, 6)
        
        if np.random.random() < 0.35:
            earned = max(3, int(attempted * np.random.uniform(0.4, 0.8)))
        else:
            earned = attempted
        
        semesters.append({'credits_attempted': attempted, 'credits_earned': earned})
    
    return semesters, archetype

def generate_super_achiever():
    semesters = []
    archetype = "super_achiever"
    credit_pattern = [22, 24, 22, 21, 19, 16, 13, 11]
    
    for sem in range(8):
        attempted = earned = credit_pattern[sem] + np.random.randint(-2, 3)
        earned = max(earned, 12)
        semesters.append({'credits_attempted': attempted, 'credits_earned': earned})
    
    return semesters, archetype

def generate_inconsistent_student():
    semesters = []
    archetype = "inconsistent"
    
    for sem in range(8):
        attempted = np.random.randint(9, 24)
        if np.random.random() < 0.28:
            earned = max(6, int(attempted * np.random.uniform(0.6, 0.9)))
        else:
            earned = attempted
        semesters.append({'credits_attempted': attempted, 'credits_earned': earned})
    
    return semesters, archetype

def generate_late_bloomer():
    semesters = []
    archetype = "late_bloomer"
    credit_pattern = [10, 12, 14, 16, 17, 18, 19, 20]
    
    for sem in range(8):
        attempted = earned = credit_pattern[sem] + np.random.randint(-2, 3)
        earned = max(earned, 8)
        semesters.append({'credits_attempted': attempted, 'credits_earned': earned})
    
    return semesters, archetype

def generate_early_struggler():
    semesters = []
    archetype = "early_struggler"
    credit_pattern = [8, 10, 12, 15, 17, 18, 19, 20]
    
    for sem in range(8):
        attempted = credit_pattern[sem] + np.random.randint(-1, 3)
        
        if sem < 3 and np.random.random() < 0.4:
            earned = max(3, int(attempted * np.random.uniform(0.4, 0.8)))
        else:
            earned = attempted
        
        semesters.append({'credits_attempted': attempted, 'credits_earned': earned})
    
    return semesters, archetype

def generate_burnout_student():
    semesters = []
    archetype = "burnout"
    credit_pattern = [20, 21, 19, 18, 16, 15, 13, 12]
    
    for sem in range(8):
        attempted = credit_pattern[sem] + np.random.randint(-2, 2)
        attempted = max(attempted, 9)
        
        if sem > 4 and np.random.random() < 0.22:
            earned = max(6, int(attempted * np.random.uniform(0.7, 0.9)))
        else:
            earned = attempted
        
        semesters.append({'credits_attempted': attempted, 'credits_earned': earned})
    
    return semesters, archetype

def generate_part_time_student():
    semesters = []
    archetype = "part_time"
    
    for sem in range(8):
        attempted = earned = np.random.randint(9, 15)
        semesters.append({'credits_attempted': attempted, 'credits_earned': earned})
    
    return semesters, archetype

def generate_overloader_student():
    semesters = []
    archetype = "overloader"
    
    for sem in range(8):
        attempted = np.random.randint(18, 26)
        
        if np.random.random() < 0.18:
            earned = max(15, int(attempted * np.random.uniform(0.8, 0.95)))
        else:
            earned = attempted
        
        semesters.append({'credits_attempted': attempted, 'credits_earned': earned})
    
    return semesters, archetype

def generate_repeat_courses_journey():
    semesters = []
    archetype = "repeat_courses"
    
    for sem in range(8):
        attempted = np.random.randint(12, 19)
        fail_rate = max(0.1, 0.42 - sem * 0.04)
        
        if np.random.random() < fail_rate:
            earned = max(6, int(attempted * np.random.uniform(0.5, 0.8)))
        else:
            earned = attempted
        
        semesters.append({'credits_attempted': attempted, 'credits_earned': earned})
    
    return semesters, archetype

def generate_gap_semester_journey():
    semesters = []
    archetype = "gap_semesters"
    gap_semesters = np.random.choice([3, 5, 6], size=np.random.randint(1, 3), replace=False)
    
    for sem in range(8):
        if (sem + 1) in gap_semesters:
            semesters.append({'credits_attempted': 0, 'credits_earned': 0})
        else:
            attempted = earned = np.random.randint(12, 20)
            semesters.append({'credits_attempted': attempted, 'credits_earned': earned})
    
    return semesters, archetype

def generate_working_student_journey():
    semesters = []
    archetype = "working_student"
    
    for sem in range(8):
        if np.random.random() < 0.38:
            attempted = earned = np.random.randint(6, 12)
        else:
            attempted = earned = np.random.randint(13, 18)
        
        semesters.append({'credits_attempted': attempted, 'credits_earned': earned})
    
    return semesters, archetype

def generate_transfer_journey():
    semesters = []
    archetype = "transfer_student"
    
    for sem in range(8):
        attempted = np.random.randint(14, 19)
        
        if sem == 3:  # Credit loss during transfer
            earned = max(6, int(attempted * np.random.uniform(0.4, 0.7)))
        else:
            earned = attempted
        
        semesters.append({'credits_attempted': attempted, 'credits_earned': earned})
    
    return semesters, archetype

def generate_normal_student_with_noise():
    semesters = []
    archetype = "normal"
    
    for sem in range(8):
        attempted = np.random.randint(13, 19)
        
        if np.random.random() < 0.12:
            earned = max(9, int(attempted * np.random.uniform(0.75, 0.95)))
        else:
            earned = attempted
        
        semesters.append({'credits_attempted': attempted, 'credits_earned': earned})
    
    return semesters, archetype

def generate_yearly_gpa_fluctuations(student_id, base_gpa, archetype):
    """Generate GPA effects over 4 academic years"""
    student_seed = hash(student_id + "_yearly_trends") % (2**32)
    np.random.seed(student_seed)
    
    # Year 1: Adjustment
    year1_effect = np.random.normal(0, 0.3)
    if archetype in ['early_struggler', 'struggling']:
        year1_effect -= 0.4
    elif archetype == 'super_achiever':
        year1_effect += 0.2
    
    # Year 2: Settling
    year2_trend = np.random.choice([-0.2, -0.1, 0, 0.1, 0.2], p=[0.1, 0.2, 0.4, 0.2, 0.1])
    if archetype == 'late_bloomer':
        year2_trend += 0.3
    elif archetype == 'burnout':
        year2_trend -= 0.1
    
    # Year 3: Mid-program
    year3_trend = np.random.choice([-0.3, -0.1, 0, 0.1, 0.3], p=[0.15, 0.25, 0.3, 0.2, 0.1])
    if archetype == 'late_bloomer':
        year3_trend += 0.4
    elif archetype == 'burnout':
        year3_trend -= 0.3
    elif archetype == 'inconsistent':
        year3_trend = np.random.normal(0, 0.5)
    
    # Year 4: Final push
    year4_trend = np.random.choice([-0.2, 0, 0.2, 0.3], p=[0.2, 0.3, 0.3, 0.2])
    if archetype == 'burnout':
        year4_trend -= 0.4
    elif archetype in ['late_bloomer', 'early_struggler']:
        year4_trend += 0.3
    
    return [year1_effect, year2_trend, year3_trend, year4_trend]

def generate_semester_gpa_with_all_effects(student_id, archetype, journey, enrollment_year, 
                                          institutional_trends, gender_effects):
    """Generate GPA with time-series, institutional, and gender effects"""
    student_seed = hash(student_id + "_gpa_series") % (2**32)
    np.random.seed(student_seed)
    
    gender, gpa_modifier, _, _, _ = gender_effects
    
    # Base GPA by archetype
    base_gpa_ranges = {
        'struggling': (1.8, 2.4), 'repeat_courses': (2.0, 2.8), 'early_struggler': (2.0, 2.6),
        'part_time': (2.8, 3.3), 'working_student': (2.4, 3.2), 'normal': (2.6, 3.6),
        'gap_semesters': (2.5, 3.4), 'transfer_student': (2.6, 3.4), 'inconsistent': (2.8, 3.4),
        'late_bloomer': (2.2, 2.8), 'overloader': (3.1, 3.6), 'super_achiever': (3.6, 3.9),
        'burnout': (3.5, 3.9)
    }
    
    gpa_min, gpa_max = base_gpa_ranges.get(archetype, (2.6, 3.6))
    base_gpa = np.random.uniform(gpa_min, gpa_max)
    
    # Apply gender and institutional effects
    base_gpa += gpa_modifier
    institutional_gpa_effect, _, _ = institutional_trends.get(enrollment_year, (0, 0.01, 0.03))
    base_gpa += institutional_gpa_effect
    
    # Get yearly fluctuations
    yearly_effects = generate_yearly_gpa_fluctuations(student_id, base_gpa, archetype)
    
    semester_gpas = []
    
    for semester in range(8):
        academic_year = semester // 2
        yearly_effect = yearly_effects[academic_year]
        semester_variation = np.random.normal(0, 0.15)
        
        # Archetype-specific patterns
        if archetype == 'inconsistent':
            semester_variation = np.random.normal(0, 0.4)
        elif archetype == 'late_bloomer':
            improvement = semester * 0.08
            semester_variation += improvement
        elif archetype == 'burnout':
            decline = semester * -0.06
            semester_variation += decline
        elif archetype == 'early_struggler':
            if semester >= 2:
                improvement = (semester - 2) * 0.1
                semester_variation += improvement
        
        semester_gpa = base_gpa + yearly_effect + semester_variation
        
        # Course failure effects
        sem_data = journey[semester]
        if sem_data['credits_earned'] < sem_data['credits_attempted']:
            failure_rate = 1 - (sem_data['credits_earned'] / max(sem_data['credits_attempted'], 1))
            gpa_penalty = failure_rate * 1.5
            semester_gpa -= gpa_penalty
        
        semester_gpa = np.clip(semester_gpa, 0.0, 4.0)
        semester_gpas.append(round(semester_gpa, 2))
    
    return semester_gpas

def check_dropout_probability_with_all_effects(student_id, semester, archetype, cumulative_gpa, 
                                             cumulative_credits, enrollment_year, 
                                             institutional_trends, gender_effects):
    """Check dropout with all effects"""
    student_seed = hash(student_id + f"_dropout_{semester}") % (2**32)
    np.random.seed(student_seed)
    
    _, _, _, dropout_modifier, _ = gender_effects
    
    base_dropout_rate = 0.002
    
    archetype_multipliers = {
        'struggling': 8.0, 'repeat_courses': 6.0, 'inconsistent': 4.0,
        'early_struggler': 3.0, 'burnout': 3.5, 'gap_semesters': 2.0,
        'transfer_student': 1.5, 'working_student': 2.0, 'part_time': 1.2,
        'normal': 1.0, 'late_bloomer': 0.8, 'overloader': 0.6, 'super_achiever': 0.3
    }
    
    if cumulative_gpa < 2.0:
        gpa_multiplier = 5.0
    elif cumulative_gpa < 2.5:
        gpa_multiplier = 3.0
    elif cumulative_gpa < 3.0:
        gpa_multiplier = 1.5
    else:
        gpa_multiplier = 0.8
    
    if semester <= 2:
        semester_multiplier = 2.0
    elif semester <= 4:
        semester_multiplier = 1.5
    else:
        semester_multiplier = 0.8
    
    _, institutional_dropout_rate, _ = institutional_trends.get(enrollment_year, (0, 0.01, 0.03))
    year_multiplier = institutional_dropout_rate / 0.01
    
    archetype_mult = archetype_multipliers.get(archetype, 1.0)
    dropout_prob = (base_dropout_rate * archetype_mult * gpa_multiplier * 
                   semester_multiplier * year_multiplier * dropout_modifier)
    dropout_prob = min(dropout_prob, 0.05)
    
    return np.random.random() < dropout_prob

def check_early_graduation_eligibility_with_all_effects(cumulative_credits, semester, archetype, 
                                                      enrollment_year, institutional_trends, 
                                                      gender_effects):
    """Check early graduation with all effects"""
    if cumulative_credits < 132 or semester >= 8:
        return False
    
    _, _, early_grad_modifier, _, _ = gender_effects
    
    graduation_probabilities = {
        'super_achiever': 0.7, 'overloader': 0.5, 'normal': 0.3,
        'late_bloomer': 0.4, 'early_struggler': 0.6, 'inconsistent': 0.2,
        'burnout': 0.8, 'part_time': 0.1, 'struggling': 0.9,
        'repeat_courses': 0.7, 'gap_semesters': 0.4, 'working_student': 0.6,
        'transfer_student': 0.5
    }
    
    base_prob = graduation_probabilities.get(archetype, 0.3)
    
    _, _, institutional_early_grad_rate = institutional_trends.get(enrollment_year, (0, 0.01, 0.03))
    year_multiplier = institutional_early_grad_rate / 0.03
    
    adjusted_prob = base_prob * year_multiplier * early_grad_modifier
    adjusted_prob = min(adjusted_prob, 0.9)
    
    return np.random.random() < adjusted_prob

def apply_gender_effects_to_credits(credits_attempted, gender_effects):
    """Apply gender effects to credit loads"""
    _, _, _, _, credit_load_modifier = gender_effects
    
    adjusted_credits = int(credits_attempted * credit_load_modifier * np.random.normal(1.0, 0.05))
    adjusted_credits = max(adjusted_credits, 6)
    adjusted_credits = min(adjusted_credits, 24)
    
    return adjusted_credits

def generate_complete_synthetic_data():
    """
    Generate complete synthetic dataset: 2001-2025, 800-1200 students/year, 
    with gender effects and time-series
    """
    print("Generating Complete Synthetic Student Dataset...")
    print("Features:")
    print("- Years: 2001-2025 (25 years)")
    print("- Students: 800-1200 per year")
    print("- Gender effects (subtle)")
    print("- Individual time-series (4 academic years)")
    print("- Institutional trends (economic/policy effects)")
    print("- Multiple student archetypes")
    print("- Early graduation and dropout scenarios")
    print("-" * 60)
    
    institutional_trends = generate_institutional_yearly_trends()
    
    data = []
    archetypes = {}
    gender_stats = {'Male': 0, 'Female': 0}
    yearly_stats = {}
    
    np.random.seed(42)
    
    for year in range(2001, 2026):
        num_students = np.random.randint(800, 1201)
        year_dropouts = {'Male': 0, 'Female': 0}
        year_early_grads = {'Male': 0, 'Female': 0}
        year_gpas = {'Male': [], 'Female': []}
        
        print(f"Processing year {year}: {num_students} students...")
        
        for student_num in range(num_students):
            student_id = f"{year}_{student_num:04d}"
            
            # Generate gender and effects
            gender_effects = generate_gender_and_effects(student_id)
            gender = gender_effects[0]
            gender_stats[gender] += 1
            
            # Generate student profile
            journey, archetype = generate_student_performance_profile(student_id, year)
            
            # Apply gender effects to credit loads
            for sem_data in journey:
                original_attempted = sem_data['credits_attempted']
                adjusted_attempted = apply_gender_effects_to_credits(original_attempted, gender_effects)
                sem_data['credits_attempted'] = adjusted_attempted
                if sem_data['credits_earned'] > adjusted_attempted:
                    sem_data['credits_earned'] = adjusted_attempted
            
            if archetype not in archetypes:
                archetypes[archetype] = 0
            archetypes[archetype] += 1
            
            # Generate GPA with all effects
            semester_gpas = generate_semester_gpa_with_all_effects(
                student_id, archetype, journey, year, institutional_trends, gender_effects
            )
            
            # Process semester records
            cumulative_credits = 0
            cumulative_grade_points = 0
            graduated_early = False
            dropped_out = False
            final_cumulative_credits = 0
            final_cumulative_gpa = 0.0
            
            for semester, sem_data in enumerate(journey, 1):
                if graduated_early or dropped_out:
                    record = {
                        'student_id': student_id,
                        'enrollment_year': year,
                        'gender': gender,
                        'semester': semester,
                        'academic_year': (semester + 1) // 2,
                        'semester_in_year': 'Fall' if semester % 2 == 1 else 'Spring',
                        'semester_year': year + (semester - 1) // 2,
                        'major': 'SE',
                        'credits_attempted': 0,
                        'credits_earned': 0,
                        'gpa': 0.00,
                        'cumulative_credits': final_cumulative_credits,
                        'cumulative_gpa': final_cumulative_gpa,
                        'status': 'graduated' if graduated_early else 'dropped_out'
                    }
                    data.append(record)
                    continue
                
                # Normal semester processing
                credits_attempted = sem_data['credits_attempted']
                credits_earned = sem_data['credits_earned']
                semester_gpa = semester_gpas[semester - 1]
                
                # Update cumulatives
                if credits_earned > 0 and semester_gpa > 0:
                    cumulative_credits += credits_earned
                    cumulative_grade_points += semester_gpa * credits_earned
                    cumulative_gpa = round(cumulative_grade_points / cumulative_credits, 3)
                else:
                    cumulative_gpa = round(cumulative_grade_points / cumulative_credits, 3) if cumulative_credits > 0 else 0.0
                
                # Record semester
                record = {
                    'student_id': student_id,
                    'enrollment_year': year,
                    'gender': gender,
                    'semester': semester,
                    'academic_year': (semester + 1) // 2,
                    'semester_in_year': 'Fall' if semester % 2 == 1 else 'Spring',
                    'semester_year': year + (semester - 1) // 2,
                    'major': 'SE',
                    'credits_attempted': credits_attempted,
                    'credits_earned': credits_earned,
                    'gpa': semester_gpa,
                    'cumulative_credits': cumulative_credits,
                    'cumulative_gpa': cumulative_gpa,
                    'status': 'active'
                }
                data.append(record)
                year_gpas[gender].append(semester_gpa)
                
                # Check dropout
                if (semester < 8 and 
                    check_dropout_probability_with_all_effects(
                        student_id, semester, archetype, cumulative_gpa, 
                        cumulative_credits, year, institutional_trends, gender_effects)):
                    dropped_out = True
                    final_cumulative_credits = cumulative_credits
                    final_cumulative_gpa = cumulative_gpa
                    year_dropouts[gender] += 1
                    data[-1]['status'] = 'dropped_out'
                    continue
                
                # Check early graduation
                if (semester < 8 and cumulative_credits >= 132 and 
                    check_early_graduation_eligibility_with_all_effects(
                        cumulative_credits, semester, archetype, year, 
                        institutional_trends, gender_effects)):
                    graduated_early = True
                    final_cumulative_credits = cumulative_credits
                    final_cumulative_gpa = cumulative_gpa
                    year_early_grads[gender] += 1
                    data[-1]['status'] = 'graduated'
        
        # Store yearly statistics
        yearly_stats[year] = {
            'students': num_students,
            'dropouts': year_dropouts,
            'early_graduates': year_early_grads,
            'avg_gpa': {gender: np.mean(gpas) if gpas else 0.0 for gender, gpas in year_gpas.items()}
        }
    
    df = pd.DataFrame(data)
    
    return df, yearly_stats, archetypes, institutional_trends, gender_stats

def show_comprehensive_analysis(df, yearly_stats, gender_stats):
    """Show comprehensive dataset analysis"""
    print("\n" + "="*80)
    print("COMPREHENSIVE DATASET ANALYSIS")
    print("="*80)
    
    total_students = sum(gender_stats.values())
    total_records = len(df)
    years_covered = df['enrollment_year'].nunique()
    
    print(f"Dataset Overview:")
    print(f"  Total records: {total_records:,}")
    print(f"  Total students: {total_students:,}")
    print(f"  Years covered: {years_covered} (2001-2025)")
    print(f"  Average students/year: {total_students/years_covered:.0f}")
    print(f"  Records per student: {total_records/total_students:.0f}")
    
    print(f"\nGender Distribution:")
    for gender, count in gender_stats.items():
        print(f"  {gender}: {count:,} ({count/total_students*100:.1f}%)")
    
    # Outcome analysis
    student_outcomes = []
    for student_id in df['student_id'].unique():
        student_data = df[df['student_id'] == student_id]
        gender = student_data['gender'].iloc[0]
        
        if 'dropped_out' in student_data['status'].values:
            outcome = 'dropped_out'
        elif 'graduated' in student_data['status'].values:
            sem8_data = student_data[student_data['semester'] == 8]
            if (len(sem8_data) > 0 and 
                sem8_data.iloc[0]['credits_attempted'] == 0 and 
                sem8_data.iloc[0]['cumulative_credits'] >= 132):
                outcome = 'early_graduate'
            else:
                outcome = 'regular_graduate'
        else:
            outcome = 'regular_graduate'
        
        student_outcomes.append({'student_id': student_id, 'gender': gender, 'outcome': outcome})
    
    outcomes_df = pd.DataFrame(student_outcomes)
    
    print(f"\nOverall Outcomes:")
    outcome_counts = outcomes_df['outcome'].value_counts()
    for outcome, count in outcome_counts.items():
        print(f"  {outcome}: {count:,} ({count/total_students*100:.1f}%)")
    # Gender-based outcome analysis
    print(f"\nOutcomes by Gender:")
    gender_outcomes = outcomes_df.groupby(['gender', 'outcome']).size().unstack(fill_value=0)
    
    for col in ['regular_graduate', 'early_graduate', 'dropped_out']:
        if col not in gender_outcomes.columns:
            gender_outcomes[col] = 0
    
    gender_outcomes['total'] = gender_outcomes.sum(axis=1)
    gender_outcomes['dropout_rate'] = (gender_outcomes['dropped_out'] / gender_outcomes['total'] * 100).round(1)
    gender_outcomes['early_grad_rate'] = (gender_outcomes['early_graduate'] / gender_outcomes['total'] * 100).round(1)
    
    print(gender_outcomes[['dropped_out', 'early_graduate', 'regular_graduate', 'dropout_rate', 'early_grad_rate']].to_string())
    
    # Time trends
    print(f"\nTime Trends (First vs Last 3 years):")
    early_years = [2001, 2002, 2003]
    late_years = [2023, 2024, 2025]
    
    early_data = df[df['enrollment_year'].isin(early_years)]
    late_data = df[df['enrollment_year'].isin(late_years)]
    
    print(f"2001-2003 avg GPA: {early_data['gpa'].mean():.3f}")
    print(f"2023-2025 avg GPA: {late_data['gpa'].mean():.3f}")
    print(f"GPA change: {late_data['gpa'].mean() - early_data['gpa'].mean():+.3f}")

def show_sample_student_journeys(df):
    """Show examples of complete student journeys"""
    print("\n" + "="*100)
    print("SAMPLE STUDENT ACADEMIC JOURNEYS")
    print("="*100)
    
    # Show 3 different types of students
    sample_students = df['student_id'].unique()[:3]
    
    for i, student_id in enumerate(sample_students, 1):
        student_data = df[df['student_id'] == student_id]
        gender = student_data['gender'].iloc[0]
        enrollment_year = student_data['enrollment_year'].iloc[0]
        
        print(f"\n--- STUDENT {i}: {student_id} ({gender}, Enrolled {enrollment_year}) ---")
        
        # Show progression
        display_cols = ['semester', 'academic_year', 'semester_in_year', 'credits_attempted', 
                       'credits_earned', 'gpa', 'cumulative_credits', 'cumulative_gpa', 'status']
        print(student_data[display_cols].to_string(index=False))
        
        # Summary
        final_record = student_data.iloc[-1]
        print(f"Final: {final_record['cumulative_credits']} credits, {final_record['cumulative_gpa']:.3f} GPA, Status: {final_record['status']}")

def show_institutional_trends_over_time(yearly_stats):
    """Show how institutional factors changed over 25 years"""
    print("\n" + "="*80)
    print("INSTITUTIONAL TRENDS (2001-2025)")
    print("="*80)
    
    years = sorted(yearly_stats.keys())
    
    # Show key milestone years
    milestone_years = [2001, 2008, 2012, 2020, 2025]
    
    print("Key Milestone Years:")
    print("Year | Students | Male Avg GPA | Female Avg GPA | Dropout Rate | Early Grad Rate")
    print("-" * 75)
    
    for year in milestone_years:
        if year in yearly_stats:
            stats = yearly_stats[year]
            male_gpa = stats['avg_gpa'].get('Male', 0)
            female_gpa = stats['avg_gpa'].get('Female', 0)
            
            total_dropouts = stats['dropouts']['Male'] + stats['dropouts']['Female']
            total_early_grads = stats['early_graduates']['Male'] + stats['early_graduates']['Female']
            
            dropout_rate = total_dropouts / stats['students'] * 100
            early_grad_rate = total_early_grads / stats['students'] * 100
            
            print(f"{year} |   {stats['students']:4d}   |    {male_gpa:5.2f}     |     {female_gpa:5.2f}      |    {dropout_rate:4.1f}%    |     {early_grad_rate:4.1f}%")
    
    # Show overall trends
    first_year_stats = yearly_stats[2001]
    last_year_stats = yearly_stats[2025]
    
    print(f"\n25-Year Trends:")
    first_male_gpa = first_year_stats['avg_gpa'].get('Male', 0)
    last_male_gpa = last_year_stats['avg_gpa'].get('Male', 0)
    first_female_gpa = first_year_stats['avg_gpa'].get('Female', 0)
    last_female_gpa = last_year_stats['avg_gpa'].get('Female', 0)
    
    print(f"Male GPA: {first_male_gpa:.3f} → {last_male_gpa:.3f} ({last_male_gpa - first_male_gpa:+.3f})")
    print(f"Female GPA: {first_female_gpa:.3f} → {last_female_gpa:.3f} ({last_female_gpa - first_female_gpa:+.3f})")

if __name__ == "__main__":
    np.random.seed(42)
    
    print("Starting complete synthetic data generation...")
    print("This may take a few minutes due to the large dataset size...")
    
    # Generate the complete dataset
    df, yearly_stats, archetypes, institutional_trends, gender_stats = generate_complete_synthetic_data()
    
    
    # Save the complete dataset
    print(f"\nSaving dataset...")
    output_filename = 'complete_synthetic_student_data_2001_2025.csv'
    df.to_csv(output_filename, index=False)
    
    print(f"\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"File saved: {output_filename}")
    print(f"Dataset size: {df.shape}")
    print(f"File size: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    print(f"\nDataset Features:")
    print(f"- 25 years of data (2001-2025)")
    print(f"- {len(df['student_id'].unique()):,} total students")
    print(f"- Gender effects (subtle)")
    print(f"- Individual time-series over 4 academic years")
    print(f"- Institutional year-over-year trends")
    print(f"- 13 student archetypes with realistic patterns")
    print(f"- Early graduation and dropout scenarios")
    print(f"- Realistic GPA, credit, and outcome distributions")
    
    print(f"\nColumns in dataset:")
    for col in df.columns:
        print(f"  - {col}")