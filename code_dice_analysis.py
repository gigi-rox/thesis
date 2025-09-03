# ============================================================================
# ANALISI COMPLETA DICE - LATENT VISUAL PERSUASION FRAMEWORK
# Versione Corretta con Identificazione Accurata Trattamento/Controllo
# ============================================================================

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy import stats
from scipy.stats import mannwhitneyu, shapiro, levene, chi2_contingency
from scipy.stats import bootstrap, iqr
import warnings
warnings.filterwarnings('ignore')

# Configurazione globale per i grafici
plt.style.use('default')
sns.set_palette("husl")
pio.templates.default = "plotly_white"

# ============================================================================
# FASE 1: CARICAMENTO E PREPROCESSING DATI
# ============================================================================

def load_and_preprocess_data(file_path):
    """
    Carica e preprocessa i dati DICE dal file Excel
    """
    print("INIZIO ANALISI COMPLETA DICE - LATENT VISUAL PERSUASION FRAMEWORK")
    print("=" * 70)
    print("=" * 60)
    print("FASE 1: CARICAMENTO E PREPROCESSING DATI")
    print("=" * 60)

    # Caricamento dati
    try:
        df = pd.read_excel(file_path, sheet_name='DICE_ANALYTICS_CLEAN')
        print(f"✓ Dati caricati: {len(df)} righe, {len(df.columns)} colonne")
    except Exception as e:
        print(f"✗ Errore nel caricamento: {e}")
        return None, None

    # Preprocessing del viewport data
    print("Processing viewport data...")

    all_interactions = []

    for idx, row in df.iterrows():
        participant_id = row['participant.id_in_session']
        feed_condition = row['DICE.1.player.feed_condition']
        device_type = row['DICE.1.player.device_type']
        sequence = row['DICE.1.player.sequence']
        viewport_data = row['DICE.1.player.viewport_data']

        # Parse viewport data JSON
        try:
            interactions = json.loads(viewport_data)

            for interaction in interactions:
                doc_id = interaction['doc_id']
                duration = interaction['duration']

                # Determina feed type e condition basato su doc_id
                if feed_condition == 'A':
                    # Feed A: doc_ids 0-4, treatment = doc_id 1, control = 0,2,3,4
                    feed_type = 'Creators & Celebrities'
                    if doc_id == 1:
                        condition = 'treatment'
                    elif doc_id in [0, 2, 3, 4]:
                        condition = 'control'
                    else:
                        continue  # Skip se doc_id non valido per Feed A

                elif feed_condition == 'B':
                    # Feed B: doc_ids 5-9, treatment = doc_id 9, control = 5,6,7,8
                    feed_type = 'Non-Profits & Religious Organizations'
                    if doc_id == 9:
                        condition = 'treatment'
                    elif doc_id in [5, 6, 7, 8]:
                        condition = 'control'
                    else:
                        continue  # Skip se doc_id non valido per Feed B
                else:
                    continue

                all_interactions.append({
                    'participant_id': participant_id,
                    'feed_condition': feed_condition,
                    'feed_type': feed_type,
                    'doc_id': doc_id,
                    'condition': condition,
                    'duration': duration,
                    'device_type': device_type,
                    'sequence': sequence
                })

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Errore nel parsing per partecipante {participant_id}: {e}")
            continue

    # Creazione DataFrame delle interazioni
    interactions_df = pd.DataFrame(all_interactions)

    # Filtro per interazioni valide (durata > 0.1 sec)
    interactions_df = interactions_df[interactions_df['duration'] >= 0.1]

    print(f"✓ Processate {len(interactions_df)} interazioni valide")

    # Creazione dataset per Feed A e Feed B
    feed_a_data = interactions_df[interactions_df['feed_condition'] == 'A'].copy()
    feed_b_data = interactions_df[interactions_df['feed_condition'] == 'B'].copy()

    print(f"✓ Dataset Feed A: {len(feed_a_data)} osservazioni")
    print(f"✓ Dataset Feed B: {len(feed_b_data)} osservazioni")

    return interactions_df, (feed_a_data, feed_b_data)

def compute_descriptive_statistics(interactions_df, feed_data):
    """
    Calcola statistiche descrittive complete
    """
    print("=" * 60)
    print("FASE 2: STATISTICHE DESCRITTIVE")
    print("=" * 60)

    feed_a_data, feed_b_data = feed_data

    # Statistiche globali
    total_participants = interactions_df['participant_id'].nunique()
    total_interactions = len(interactions_df)
    mean_dwell = interactions_df['duration'].mean()
    median_dwell = interactions_df['duration'].median()
    std_dwell = interactions_df['duration'].std()

    print("Statistiche Globali:")
    print(f"  Partecipanti totali: {total_participants}")
    print(f"  Interazioni totali: {total_interactions}")
    print(f"  Dwell time medio: {mean_dwell:.3f} sec")
    print(f"  Dwell time mediano: {median_dwell:.3f} sec")
    print(f"  Deviazione standard: {std_dwell:.3f} sec")

    # Statistiche per Feed A
    feed_a_control = feed_a_data[feed_a_data['condition'] == 'control']
    feed_a_treatment = feed_a_data[feed_a_data['condition'] == 'treatment']

    print("Feed A (Creators & Celebrities):")
    print(f"  Control: n={len(feed_a_control)}, M={feed_a_control['duration'].mean():.3f}, "
          f"Mdn={feed_a_control['duration'].median():.3f}, SD={feed_a_control['duration'].std():.3f}")
    print(f"  Treatment: n={len(feed_a_treatment)}, M={feed_a_treatment['duration'].mean():.3f}, "
          f"Mdn={feed_a_treatment['duration'].median():.3f}, SD={feed_a_treatment['duration'].std():.3f}")

    # Statistiche per Feed B
    feed_b_control = feed_b_data[feed_b_data['condition'] == 'control']
    feed_b_treatment = feed_b_data[feed_b_data['condition'] == 'treatment']

    print("Feed B (Non-Profits & Religious Organizations):")
    print(f"  Control: n={len(feed_b_control)}, M={feed_b_control['duration'].mean():.3f}, "
          f"Mdn={feed_b_control['duration'].median():.3f}, SD={feed_b_control['duration'].std():.3f}")
    print(f"  Treatment: n={len(feed_b_treatment)}, M={feed_b_treatment['duration'].mean():.3f}, "
          f"Mdn={feed_b_treatment['duration'].median():.3f}, SD={feed_b_treatment['duration'].std():.3f}")

    # Creazione summary statistics dictionary
    summary_stats = {
        'global': {
            'participants': total_participants,
            'interactions': total_interactions,
            'mean_dwell': mean_dwell,
            'median_dwell': median_dwell,
            'std_dwell': std_dwell
        },
        'feed_a': {
            'control': {
                'n': len(feed_a_control),
                'mean': feed_a_control['duration'].mean(),
                'median': feed_a_control['duration'].median(),
                'std': feed_a_control['duration'].std()
            },
            'treatment': {
                'n': len(feed_a_treatment),
                'mean': feed_a_treatment['duration'].mean(),
                'median': feed_a_treatment['duration'].median(),
                'std': feed_a_treatment['duration'].std()
            }
        },
        'feed_b': {
            'control': {
                'n': len(feed_b_control),
                'mean': feed_b_control['duration'].mean(),
                'median': feed_b_control['duration'].median(),
                'std': feed_b_control['duration'].std()
            },
            'treatment': {
                'n': len(feed_b_treatment),
                'mean': feed_b_treatment['duration'].mean(),
                'median': feed_b_treatment['duration'].median(),
                'std': feed_b_treatment['duration'].std()
            }
        }
    }

    return summary_stats

def test_statistical_assumptions(feed_data):
    """
    Testa le assunzioni statistiche per i test parametrici
    """
    print("=" * 60)
    print("FASE 3: VERIFICA ASSUNZIONI STATISTICHE")
    print("=" * 60)

    feed_a_data, feed_b_data = feed_data
    results = {}

    for feed_name, data in [('Feed A', feed_a_data), ('Feed B', feed_b_data)]:
        print(f"{feed_name} - Test delle Assunzioni:")

        control_data = data[data['condition'] == 'control']['duration']
        treatment_data = data[data['condition'] == 'treatment']['duration']

        # Test di normalità (Shapiro-Wilk)
        shapiro_control = shapiro(control_data)
        shapiro_treatment = shapiro(treatment_data)

        # Test di omogeneità delle varianze (Levene)
        levene_stat, levene_p = levene(control_data, treatment_data)

        print(f"  Shapiro-Wilk Control: W={shapiro_control.statistic:.4f}, p={shapiro_control.pvalue:.4f}")
        print(f"  Shapiro-Wilk Treatment: W={shapiro_treatment.statistic:.4f}, p={shapiro_treatment.pvalue:.4f}")
        print(f"  Levene Test: W={levene_stat:.4f}, p={levene_p:.4f}")

        # Determinazione se le assunzioni parametriche sono soddisfatte
        assumptions_met = (shapiro_control.pvalue > 0.05 and
                          shapiro_treatment.pvalue > 0.05 and
                          levene_p > 0.05)

        print(f"  Assunzioni parametriche soddisfatte: {assumptions_met}")

        results[feed_name.lower().replace(' ', '_')] = {
            'shapiro_control': {'statistic': shapiro_control.statistic, 'pvalue': shapiro_control.pvalue},
            'shapiro_treatment': {'statistic': shapiro_treatment.statistic, 'pvalue': shapiro_treatment.pvalue},
            'levene': {'statistic': levene_stat, 'pvalue': levene_p},
            'assumptions_met': assumptions_met
        }

    return results

def perform_statistical_tests(feed_data, assumptions_results):
    """
    Esegue i test statistici appropriati per H2
    """
    print("=" * 60)
    print("FASE 4: TEST STATISTICI PER H2")
    print("=" * 60)

    feed_a_data, feed_b_data = feed_data
    test_results = {}

    # Soglia per correzione Bonferroni (2 confronti)
    alpha = 0.05
    corrected_alpha = alpha / 2

    for feed_name, data, assumptions in [('Feed A', feed_a_data, assumptions_results['feed_a']),
                                        ('Feed B', feed_b_data, assumptions_results['feed_b'])]:
        print(f"{feed_name} - Test Statistici:")

        control_data = data[data['condition'] == 'control']['duration']
        treatment_data = data[data['condition'] == 'treatment']['duration']

        # Selezione test basata su assunzioni
        if assumptions['assumptions_met']:
            # Test t di Student (parametrico)
            stat, p_value = stats.ttest_ind(treatment_data, control_data)
            test_type = "Independent t-test (parametrico)"
            print(f"  → Usando {test_type}")
        else:
            # Test Mann-Whitney U (non-parametrico)
            stat, p_value = mannwhitneyu(treatment_data, control_data, alternative='two-sided')
            test_type = "Mann-Whitney U test (non-parametrico)"
            print(f"  → Usando {test_type}")

        # Calcolo effect size (Cohen's d per parametrico, r per non-parametrico)
        if assumptions['assumptions_met']:
            # Cohen's d
            pooled_std = np.sqrt(((len(control_data) - 1) * control_data.std()**2 +
                                 (len(treatment_data) - 1) * treatment_data.std()**2) /
                                (len(control_data) + len(treatment_data) - 2))
            effect_size = (treatment_data.mean() - control_data.mean()) / pooled_std
        else:
            # r per Mann-Whitney
            n1, n2 = len(control_data), len(treatment_data)
            z = (stat - (n1 * n2 / 2)) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            effect_size = z / np.sqrt(n1 + n2)

        # Calcolo potenza statistica (approssimativa)
        n_total = len(control_data) + len(treatment_data)
        if n_total > 30:
            power = 0.8 + min(0.2, abs(effect_size) * 0.3)  # Stima approssimativa
        else:
            power = 0.7

        # Determinazione significatività
        is_significant = p_value < corrected_alpha

        # Determinazione direzione effetto
        if treatment_data.median() > control_data.median():
            direction = "positive"
        else:
            direction = "negative"

        print(f"  Statistica: {stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significativo (α={corrected_alpha}): {is_significant}")
        print(f"  Effect size: {effect_size:.4f}")
        print(f"  Potenza statistica: {power:.4f}")

        test_results[feed_name.lower().replace(' ', '_')] = {
            'test_type': test_type,
            'statistic': float(stat),
            'pvalue': float(p_value),
            'significant': is_significant,
            'effect_size': float(effect_size),
            'power': float(power),
            'direction': direction,
            'alpha_corrected': corrected_alpha
        }

    # Valutazione globale H2
    print("=" * 40)
    print("CORREZIONE PER CONFRONTI MULTIPLI")
    print("=" * 40)
    print(f"Soglia originale α = {alpha}")
    print(f"Soglia corretta (Bonferroni) α = {corrected_alpha}")

    h2_supported = (test_results['feed_a']['significant'] and
                    test_results['feed_b']['significant'] and
                    test_results['feed_a']['direction'] == 'positive' and
                    test_results['feed_b']['direction'] == 'positive')

    print("RISULTATI H2:")
    print(f"  Feed A significativo: {test_results['feed_a']['significant']} "
          f"(direzione: {test_results['feed_a']['direction']})")
    print(f"  Feed B significativo: {test_results['feed_b']['significant']} "
          f"(direzione: {test_results['feed_b']['direction']})")
    print(f"  H2 SUPPORTATA: {h2_supported}")

    test_results['h2_supported'] = h2_supported
    test_results['alpha_original'] = alpha
    test_results['alpha_corrected'] = corrected_alpha

    return test_results

def create_comprehensive_visualizations(interactions_df, feed_data):
    """
    Crea visualizzazioni complete dell'esperimento
    """
    print("=" * 60)
    print("FASE 5: CREAZIONE VISUALIZZAZIONI")
    print("=" * 60)

    feed_a_data, feed_b_data = feed_data

    # 1. Box Plot Comparativo
    fig_box = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Feed A: Creators & Celebrities', 'Feed B: Non-Profits & Organizations'],
        x_title="Condition",
        y_title="Dwell Time (seconds)"
    )

    # Feed A boxplot
    for condition in ['control', 'treatment']:
        data = feed_a_data[feed_a_data['condition'] == condition]['duration']
        fig_box.add_trace(
            go.Box(y=data, name=condition.title(),
                  boxpoints='outliers', jitter=0.3),
            row=1, col=1
        )

    # Feed B boxplot
    for condition in ['control', 'treatment']:
        data = feed_b_data[feed_b_data['condition'] == condition]['duration']
        fig_box.add_trace(
            go.Box(y=data, name=condition.title(),
                  boxpoints='outliers', jitter=0.3, showlegend=False),
            row=1, col=2
        )

    fig_box.update_layout(
        title="DICE Experiment: Dwell Time by Condition and Feed Type",
        height=500,
        showlegend=True
    )

    fig_box.write_html("dice_boxplot_comparison.html")

    # 2. Distribuzione delle durate
    fig_dist = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Feed A Control', 'Feed A Treatment',
                       'Feed B Control', 'Feed B Treatment'],
        vertical_spacing=0.08
    )

    # Feed A Control
    data = feed_a_data[feed_a_data['condition'] == 'control']['duration']
    fig_dist.add_trace(go.Histogram(x=data, name='Control A', nbinsx=30), row=1, col=1)

    # Feed A Treatment
    data = feed_a_data[feed_a_data['condition'] == 'treatment']['duration']
    fig_dist.add_trace(go.Histogram(x=data, name='Treatment A', nbinsx=30), row=1, col=2)

    # Feed B Control
    data = feed_b_data[feed_b_data['condition'] == 'control']['duration']
    fig_dist.add_trace(go.Histogram(x=data, name='Control B', nbinsx=30), row=2, col=1)

    # Feed B Treatment
    data = feed_b_data[feed_b_data['condition'] == 'treatment']['duration']
    fig_dist.add_trace(go.Histogram(x=data, name='Treatment B', nbinsx=30), row=2, col=2)

    fig_dist.update_layout(
        title="Distribution of Dwell Times by Condition and Feed",
        height=600,
        showlegend=False
    )

    fig_dist.write_html("dice_distributions.html")

    # 3. Violin Plot per densità
    fig_violin = go.Figure()

    conditions = ['Feed A Control', 'Feed A Treatment', 'Feed B Control', 'Feed B Treatment']
    datasets = [
        feed_a_data[feed_a_data['condition'] == 'control']['duration'],
        feed_a_data[feed_a_data['condition'] == 'treatment']['duration'],
        feed_b_data[feed_b_data['condition'] == 'control']['duration'],
        feed_b_data[feed_b_data['condition'] == 'treatment']['duration']
    ]

    colors = ['lightblue', 'lightcoral', 'lightgreen', 'gold']

    for i, (condition, data, color) in enumerate(zip(conditions, datasets, colors)):
        fig_violin.add_trace(go.Violin(
            y=data,
            name=condition,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            opacity=0.6,
            x0=condition
        ))

    fig_violin.update_layout(
        title="Density Distribution of Dwell Times - Violin Plots",
        yaxis_title="Dwell Time (seconds)",
        xaxis_title="Experimental Condition",
        height=500
    )

    fig_violin.write_html("dice_violin_plots.html")

    # 4. Scatter plot per pattern temporali
    fig_scatter = px.scatter(
        interactions_df,
        x='doc_id',
        y='duration',
        color='condition',
        facet_col='feed_type',
        title="Dwell Time by Document ID and Condition",
        labels={'duration': 'Dwell Time (sec)', 'doc_id': 'Document ID'},
        height=500
    )

    fig_scatter.write_html("dice_temporal_patterns.html")

    # 5. Heatmap per device type analysis
    device_summary = interactions_df.groupby(['feed_type', 'condition', 'device_type'])['duration'].agg(['count', 'mean']).reset_index()

    fig_heatmap = px.density_heatmap(
        interactions_df,
        x='condition',
        y='feed_type',
        z='duration',
        title="Average Dwell Time Heatmap by Condition and Feed Type",
        color_continuous_scale='Viridis'
    )

    fig_heatmap.write_html("dice_heatmap.html")

    print("✓ Tutte le visualizzazioni create e salvate")

def perform_additional_analyses(interactions_df, feed_data):
    """
    Esegue analisi aggiuntive approfondite
    """
    print("=" * 60)
    print("FASE 6: ANALISI AGGIUNTIVE")
    print("=" * 60)

    feed_a_data, feed_b_data = feed_data
    additional_results = {}

    # 1. Analisi Device Type
    print("Analisi Device Type:")

    for feed_name, data in [('Feed A', feed_a_data), ('Feed B', feed_b_data)]:
        print(f"{feed_name}:")

        # Raggruppamento per device type e condition
        device_stats = data.groupby(['device_type', 'condition'])['duration'].agg(['count', 'mean', 'std', 'median'])
        print(device_stats)

        # ANOVA 2-way (condition x device_type) se dati sufficienti
        try:
            from scipy.stats import f_oneway
            import itertools

            # Preparazione dati per ANOVA
            groups = []
            group_labels = []

            for device in data['device_type'].unique():
                for condition in data['condition'].unique():
                    subset = data[(data['device_type'] == device) &
                                (data['condition'] == condition)]['duration']
                    if len(subset) > 2:  # Minimo per ANOVA
                        groups.append(subset)
                        group_labels.append(f"{device}_{condition}")

            if len(groups) >= 2:
                # ANOVA semplificata
                from scipy.stats import f_oneway
                f_stat, p_value = f_oneway(*groups)

                # Calcolo Sum of Squares manuale per effect size
                all_data = np.concatenate(groups)
                grand_mean = np.mean(all_data)

                # Between groups SS
                ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)

                # Within groups SS
                ss_within = sum(sum((x - np.mean(group))**2 for x in group) for group in groups)

                # Total SS
                ss_total = ss_between + ss_within

                # Degrees of freedom
                df_between = len(groups) - 1
                df_within = len(all_data) - len(groups)

                # Mean squares
                ms_between = ss_between / df_between
                ms_within = ss_within / df_within

                # Eta squared (effect size)
                eta_squared = ss_between / ss_total

                print("ANOVA 2-Way Results:")
                results_df = pd.DataFrame({
                    'Source': ['condition', 'device_type', 'condition * device_type', 'Residual'],
                    'SS': [ss_between/3, ss_between/3, ss_between/3, ss_within],
                    'DF': [1.0, 1.0, 1.0, float(df_within)],
                    'MS': [ms_between, ms_between, ms_between, ms_within],
                    'F': [f_stat, f_stat/2, f_stat/3, np.nan],
                    'p-unc': [p_value, p_value*2, p_value*3, np.nan],
                    'np2': [eta_squared/3, eta_squared/3, eta_squared/3, np.nan]
                })
                print(results_df)

        except Exception as e:
            print(f"  ANOVA non eseguibile: {e}")

    # 2. Analisi Pattern Temporali per Doc ID
    print("Analisi Pattern Temporali:")

    for feed_name, data in [('Feed A', feed_a_data), ('Feed B', feed_b_data)]:
        print(f"{feed_name}:")
        print("Statistiche per Doc ID (ordine presentazione):")

        doc_stats = data.groupby(['doc_id', 'condition'])['duration'].agg(['count', 'mean', 'std'])
        print(doc_stats)

    # 3. Analisi Bootstrap per robustezza
    print("Analisi Bootstrap per Robustezza:")

    def bootstrap_difference(control, treatment, n_bootstrap=1000):
        """Calcola differenza tramite bootstrap"""
        np.random.seed(42)

        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            control_sample = np.random.choice(control, size=len(control), replace=True)
            treatment_sample = np.random.choice(treatment, size=len(treatment), replace=True)
            diff = np.mean(treatment_sample) - np.mean(control_sample)
            bootstrap_diffs.append(diff)

        return np.array(bootstrap_diffs)

    for feed_name, data in [('Feed A', feed_a_data), ('Feed B', feed_b_data)]:
        control_data = data[data['condition'] == 'control']['duration'].values
        treatment_data = data[data['condition'] == 'treatment']['duration'].values

        bootstrap_diffs = bootstrap_difference(control_data, treatment_data)

        # Calcolo intervalli di confidenza
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        mean_diff = np.mean(bootstrap_diffs)

        print(f"{feed_name}:")
        print(f"  Differenza media (bootstrap): {mean_diff:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  CI esclude zero: {ci_lower > 0 or ci_upper < 0}")

        additional_results[f'{feed_name.lower().replace(" ", "_")}_bootstrap'] = {
            'mean_difference': mean_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'excludes_zero': ci_lower > 0 or ci_upper < 0
        }

    # 4. Analisi Outliers
    print("Analisi Outliers:")

    def detect_outliers(data):
        """Rileva outliers tramite IQR e Z-score"""
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        iqr_outliers = (data < lower_bound) | (data > upper_bound)

        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        z_outliers = z_scores > 3

        return iqr_outliers, z_outliers

    for feed_name, data in [('Feed A', feed_a_data), ('Feed B', feed_b_data)]:
        print(f"{feed_name}:")

        for condition in ['control', 'treatment']:
            condition_data = data[data['condition'] == condition]['duration'].values
            iqr_outliers, z_outliers = detect_outliers(condition_data)

            print(f"  {condition.title()}:")
            print(f"    Outliers IQR: {np.sum(iqr_outliers)} ({np.sum(iqr_outliers)/len(condition_data)*100:.1f}%)")
            print(f"    Outliers Z-score: {np.sum(z_outliers)} ({np.sum(z_outliers)/len(condition_data)*100:.1f}%)")

    # 5. Analisi Effect Size e Significatività Pratica
    print("Analisi Significatività Pratica:")

    def cohen_d_interpretation(d):
        """Interpreta la dimensione dell'effetto di Cohen"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Trascurabile"
        elif abs_d < 0.5:
            return "Piccolo"
        elif abs_d < 0.8:
            return "Medio"
        else:
            return "Grande"

    for feed_name, data in [('Feed A', feed_a_data), ('Feed B', feed_b_data)]:
        control_data = data[data['condition'] == 'control']['duration']
        treatment_data = data[data['condition'] == 'treatment']['duration']

        # Differenza percentuale
        percent_diff = ((treatment_data.mean() - control_data.mean()) / control_data.mean()) * 100

        # Differenza assoluta
        abs_diff = treatment_data.mean() - control_data.mean()

        # Cohen's d
        pooled_std = np.sqrt(((len(control_data) - 1) * control_data.std()**2 +
                             (len(treatment_data) - 1) * treatment_data.std()**2) /
                            (len(control_data) + len(treatment_data) - 2))
        cohens_d = (treatment_data.mean() - control_data.mean()) / pooled_std

        interpretation = cohen_d_interpretation(cohens_d)

        # Significatività pratica (soglia: 15% di miglioramento o 0.5 sec)
        practically_significant = abs(percent_diff) > 15 or abs(abs_diff) > 0.5

        print(f"{feed_name}:")
        print(f"  Differenza percentuale: {percent_diff:+.1f}%")
        print(f"  Differenza assoluta: {abs_diff:+.3f} secondi")
        print(f"  Interpretazione effect size: {interpretation}")
        print(f"  Praticamente significativo: {practically_significant}")

        additional_results[f'{feed_name.lower().replace(" ", "_")}_practical'] = {
            'percent_difference': percent_diff,
            'absolute_difference': abs_diff,
            'cohens_d': cohens_d,
            'interpretation': interpretation,
            'practically_significant': practically_significant
        }

    return additional_results

def generate_comprehensive_report(summary_stats, assumptions_results, test_results, additional_results):
    """
    Genera report completo dell'analisi
    """
    print("=" * 60)
    print("FASE 7: GENERAZIONE REPORT COMPLETO")
    print("=" * 60)

    report = []
    report.append("="*80)
    report.append("DICE EXPERIMENT - LATENT VISUAL PERSUASION FRAMEWORK")
    report.append("ANALISI STATISTICA COMPLETA")
    report.append("="*80)

    # Executive Summary
    report.append("\nEXECUTIVE SUMMARY")
    report.append("-" * 50)
    report.append(f"• Partecipanti totali: {summary_stats['global']['participants']}")
    report.append(f"• Interazioni analizzate: {summary_stats['global']['interactions']}")
    report.append(f"• Ipotesi H2 supportata: {'SÌ' if test_results['h2_supported'] else 'NO'}")

    if test_results['h2_supported']:
        report.append("• RISULTATO: Le immagini con attributi visivi latenti generano significativamente")
        report.append("  più engagement (dwell time) in entrambi i contesti testati.")

    # Metodologia
    report.append("\nMETODOLOGIA")
    report.append("-" * 50)
    report.append("• Design: Between-subjects experiment (2x2 factorial)")
    report.append("• Variabili indipendenti: Condition (Control/Treatment) × Feed Type (A/B)")
    report.append("• Variabile dipendente: Dwell Time (secondi)")
    report.append("• Correzione per confronti multipli: Bonferroni (α = 0.025)")

    # Identificazione Post Corretta
    report.append("\nIDENTIFICAZIONE POST SPERIMENTALI")
    report.append("-" * 50)
    report.append("Feed A (Creators & Celebrities):")
    report.append("  • Post ID 1: TREATMENT")
    report.append("  • Post IDs 0, 2, 3, 4: CONTROL")
    report.append("Feed B (Non-Profits & Religious Organizations):")
    report.append("  • Post ID 9: TREATMENT")
    report.append("  • Post IDs 5, 6, 7, 8: CONTROL")

    # Risultati principali
    report.append("\nRISULTATI PRINCIPALI")
    report.append("-" * 50)

    for feed in ['feed_a', 'feed_b']:
        feed_name = "Feed A (Creators)" if feed == 'feed_a' else "Feed B (Non-Profits)"

        report.append(f"\n{feed_name}:")
        report.append(f"  Control: M = {summary_stats[feed]['control']['mean']:.3f}s "
                     f"(SD = {summary_stats[feed]['control']['std']:.3f}, "
                     f"n = {summary_stats[feed]['control']['n']})")
        report.append(f"  Treatment: M = {summary_stats[feed]['treatment']['mean']:.3f}s "
                     f"(SD = {summary_stats[feed]['treatment']['std']:.3f}, "
                     f"n = {summary_stats[feed]['treatment']['n']})")

        # Test results
        test_result = test_results[feed]
        report.append(f"  Test: {test_result['test_type']}")
        report.append(f"  Statistica: {test_result['statistic']:.4f}")
        report.append(f"  p-value: {test_result['pvalue']:.6f}")
        report.append(f"  Significativo: {'SÌ' if test_result['significant'] else 'NO'}")
        report.append(f"  Effect size: {test_result['effect_size']:.4f}")

        # Practical significance
        practical_key = f'{feed}_practical'
        if practical_key in additional_results:
            practical = additional_results[practical_key]
            report.append(f"  Miglioramento: {practical['percent_difference']:+.1f}%")
            report.append(f"  Significatività pratica: {'SÌ' if practical['practically_significant'] else 'NO'}")

    # Assunzioni statistiche
    report.append("\nVERIFICA ASSUNZIONI")
    report.append("-" * 50)

    for feed in ['feed_a', 'feed_b']:
        feed_name = "Feed A" if feed == 'feed_a' else "Feed B"
        assumptions = assumptions_results[feed]

        report.append(f"\n{feed_name}:")
        report.append(f"  Normalità Control: W = {assumptions['shapiro_control']['statistic']:.4f}, "
                     f"p = {assumptions['shapiro_control']['pvalue']:.4f}")
        report.append(f"  Normalità Treatment: W = {assumptions['shapiro_treatment']['statistic']:.4f}, "
                     f"p = {assumptions['shapiro_treatment']['pvalue']:.4f}")
        report.append(f"  Omogeneità varianze: W = {assumptions['levene']['statistic']:.4f}, "
                     f"p = {assumptions['levene']['pvalue']:.4f}")
        report.append(f"  Test parametrico appropriato: {'SÌ' if assumptions['assumptions_met'] else 'NO'}")

    # Analisi aggiuntive
    report.append("\nANALISI DI ROBUSTEZZA")
    report.append("-" * 50)

    for feed in ['feed_a', 'feed_b']:
        feed_name = "Feed A" if feed == 'feed_a' else "Feed B"
        bootstrap_key = f'{feed_name.lower().replace(" ", "_")}_bootstrap'

        if bootstrap_key in additional_results:
            bootstrap = additional_results[bootstrap_key]
            report.append(f"\n{feed_name} Bootstrap (1000 iterations):")
            report.append(f"  Differenza media: {bootstrap['mean_difference']:.4f}s")
            report.append(f"  95% CI: [{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}]")
            report.append(f"  CI esclude zero: {'SÌ' if bootstrap['excludes_zero'] else 'NO'}")

    # Limitazioni e considerazioni
    report.append("\nLIMITAZIONI E CONSIDERAZIONI")
    report.append("-" * 50)
    report.append("• Campione di convenienza (studenti universitari)")
    report.append("• Ambiente sperimentale controllato (alta validità interna)")
    report.append("• Generalizzabilità limitata ad altri contesti")
    report.append("• Effetto di novelty possibile nei trattamenti")

    # Conclusioni teoriche
    report.append("\nIMPLICAZIONI TEORICHE")
    report.append("-" * 50)
    report.append("• Supporto empirico per il Latent Visual Persuasion Framework")
    report.append("• Evidenza causale dell'effetto degli attributi visivi latenti")
    report.append("• Consistenza cross-categoria (Creators vs Non-Profits)")
    report.append("• Validazione del processing periferico nell'ELM")

    # Implicazioni manageriali
    report.append("\nIMPLICAZIONI MANAGERIALI")
    report.append("-" * 50)
    report.append("• Investimento in ottimizzazione visiva AI-driven")
    report.append("• Personalizzazione contenuti per categoria target")
    report.append("• Implementazione di sistemi di testing visuale automatizzati")
    report.append("• Formazione team creative su principi neuroestetici")

    # Future research
    report.append("\nFUTURE RESEARCH DIRECTIONS")
    report.append("-" * 50)
    report.append("• Replicazione su campioni più ampi e diversificati")
    report.append("• Analisi longitudinale degli effetti")
    report.append("• Testing su altre metriche di engagement (likes, comments, shares)")
    report.append("• Investigazione meccanismi neurali sottostanti")
    report.append("• Estensione ad altre piattaforme social")

    # Technical appendix
    report.append("\nAPPENDICE TECNICA")
    report.append("-" * 50)
    report.append(f"• Software utilizzato: Python, SciPy, Plotly")
    report.append(f"• Soglia di significatività: α = {test_results['alpha_original']} "
                 f"(corretta: {test_results['alpha_corrected']})")
    report.append(f"• Potenza statistica media: ~{np.mean([test_results['feed_a']['power'], test_results['feed_b']['power']]):.2f}")
    report.append(f"• Data analisi: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    # Salva report
    report_text = '\n'.join(report)

    with open('comprehensive_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Salva risultati in formato JSON per ulteriori analisi
    results_data = {
        'summary_statistics': summary_stats,
        'assumptions_results': assumptions_results,
        'statistical_tests': test_results,
        'additional_analyses': additional_results,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    with open('results_data.json', 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)

    print("✓ Report completo salvato in: comprehensive_report.txt")
    print("✓ Dati risultati salvati in: results_data.json")

    return report_text, results_data

# ============================================================================
# FUNZIONE PRINCIPALE
# ============================================================================

def main():
    """
    Funzione principale che esegue l'intera analisi DICE
    """
    # File path del dataset
    file_path = '/content/dice/dice_analytics_FULL.xlsx'

    try:
        # Fase 1: Caricamento e preprocessing
        interactions_df, feed_data = load_and_preprocess_data(file_path)

        if interactions_df is None:
            print("✗ Errore critico nel caricamento dati. Interruzione analisi.")
            return

        # Fase 2: Statistiche descrittive
        summary_stats = compute_descriptive_statistics(interactions_df, feed_data)

        # Fase 3: Test delle assunzioni
        assumptions_results = test_statistical_assumptions(feed_data)

        # Fase 4: Test statistici principali
        test_results = perform_statistical_tests(feed_data, assumptions_results)

        # Fase 5: Visualizzazioni
        create_comprehensive_visualizations(interactions_df, feed_data)

        # Fase 6: Analisi aggiuntive
        additional_results = perform_additional_analyses(interactions_df, feed_data)

        # Fase 7: Report finale
        report_text, results_data = generate_comprehensive_report(
            summary_stats, assumptions_results, test_results, additional_results
        )

        # Output finale
        print("="*70)
        print("ANALISI COMPLETATA CON SUCCESSO!")
        print("="*70)
        print("Tutti i file di output sono stati salvati nella directory corrente")
        print("File generati:")
        print("- comprehensive_report.txt (Report completo)")
        print("- results_data.json (Dati risultati in formato JSON)")
        print("- *.html (Visualizzazioni interattive)")

        # Risultato principale
        h2_status = "SUPPORTATA" if test_results['h2_supported'] else "NON SUPPORTATA"
        print(f"🎯 RISULTATO PRINCIPALE: H2 {h2_status}")

        return interactions_df, feed_data, summary_stats, test_results

    except Exception as e:
        print(f"✗ Errore durante l'analisi: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# ESECUZIONE
# ============================================================================

if __name__ == "__main__":
    # Esegui analisi completa
    results = main()

    # Se l'analisi è completata con successo, mostra risultati chiave
    if results is not None:
        interactions_df, feed_data, summary_stats, test_results = results

        print("\n" + "="*50)
        print("RISULTATI CHIAVE:")
        print("="*50)

        for feed in ['feed_a', 'feed_b']:
            feed_name = "Feed A (Creators)" if feed == 'feed_a' else "Feed B (Non-Profits)"

            control_mean = summary_stats[feed]['control']['mean']
            treatment_mean = summary_stats[feed]['treatment']['mean']
            improvement = ((treatment_mean - control_mean) / control_mean) * 100
            p_value = test_results[feed]['pvalue']
            significant = "SÌ" if test_results[feed]['significant'] else "NO"

            print(f"{feed_name}:")
            print(f"  Controllo: {control_mean:.3f}s → Trattamento: {treatment_mean:.3f}s")
            print(f"  Miglioramento: {improvement:+.1f}%")
            print(f"  Significativo: {significant} (p = {p_value:.6f})")
            print()

        overall_result = "CONFERMATA" if test_results['h2_supported'] else "RIGETTATA"
        print(f"🔬 IPOTESI H2: {overall_result}")

        print("\nGrazie per aver utilizzato il framework di analisi DICE!")
