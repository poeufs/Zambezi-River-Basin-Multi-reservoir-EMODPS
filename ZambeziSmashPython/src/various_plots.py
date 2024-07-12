from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
import os
import pandas as pd

sns.set_style()


def parallel_plots(objectives_df):

    if len(objectives_df.columns) == 3:
        file_name = 'Best_objectives_BC'

        names = ['Hydropower', 'Environment', 'Irrigation']
        units = ['TWh/year', 'Deficit (cm/sec)' + r'$^2$', 'Normalized Deficit']

    mx = []
    mn = []
    for column in names:
        mx.append(str(round(objectives_df[column].max(), 1)))
        mn.append(str(round(objectives_df[column].min(), 1)))

    objectives_df_norm = (objectives_df.max() - objectives_df) / (objectives_df.max() - objectives_df.min())
    objectives_df_norm['Name'] = "All Solutions"

    # list of dfs
    dfs_to_concat = [objectives_df_norm]

    for column in names:
        filtered_df = objectives_df_norm.loc[objectives_df_norm[column] == 1, :].copy()
        filtered_df['Name'] = "Best " + column

        dfs_to_concat.append(filtered_df)

    result_df = pd.concat(dfs_to_concat, ignore_index=True)

    fig, ax1 = plt.subplots()

    gray = '#bdbdbd'
    purple = '#7a0177'
    green = '#41ab5d'
    blue = '#1d91c0'
    yellow = '#fdaa09'
    pink = '#c51b7d'

    parallel_coordinates(result_df, 'Name', color=[gray, purple, yellow, blue], linewidth=7, alpha=.8)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=4, mode="expand", borderaxespad=1.5, fontsize=18)

    i = 0
    ax1.set_xticks(np.arange(3))

    ax1.set_xticklabels(
        [mx[i] + '\n' + '\n' + names[i] + '\n' + units[i],
         mx[i + 1] + '\n' + '\n' + names[i + 1] + '\n' + units[i + 1],
         mx[i + 2] + '\n' + '\n' + names[i + 2] + '\n' + units[i + 2]], fontsize=18)
    ax2 = ax1.twiny()
    ax2.set_xticks(np.arange(3))
    ax2.set_xticklabels([mn[i], mn[i + 1], mn[i + 2]], fontsize=18)
    ax1.get_yaxis().set_visible([])
    plt.text(1.02, 0.5, 'Direction of Preference $\\rightarrow$', {'color': '#636363', 'fontsize': 20},
             horizontalalignment='left',
             verticalalignment='center',
             rotation=90,
             clip_on=False,
             transform=plt.gca().transAxes)

    fig.set_size_inches(17.5, 9)
    plt.show()


def plot_quantities():
    plt.rcParams["font.family"] = "Myriad Pro"
    sns.set_style("whitegrid")

    input_folder = '../storage_release/'
    # input_folder_objs='../for_plots/'
    target_input_folder = '../data/'
    output_folder = '../plots/'
    delta_target = np.loadtxt(target_input_folder + 'MEF_delta.txt')
    # n_objs=3

    # copy here..
    #####################################
    feature = 'bc_policy_simulation'
    # reservoirs=['itt','kgu','kgl','ka','bg','dg','cb','mn']
    reservoirs = ['itt', 'kgu', 'kgl', 'ka', 'cb']
    title = '5_res_wKGL'
    # input_file='Zambezi_'+title+'.reference'  #'.reference'change file_name

    # data= np.loadtxt('../parallel/sets/'+feature+'/'+input_file, skiprows=0+1+2-1)
    delta_release_balance = '\n('r'$r_{CB}+Q_{Shire}-r_{Irrd7}-r_{Irrd8}-r_{Irrd9}$)'
    # res_names=['Itezhitezhi','Kafue G. Upper','Kafue G. Lower','Kariba','Batoka Gorge','Devil\'s Gorge','Cahora Bassa', 'Mphanda Nkuwa']
    res_names = ['Itezhitezhi', 'Kafue G. Upper', 'Kafue G. Lower', 'Kariba', 'Cahora Bassa']
    #####copy the segment above#########

    policies = ['best_hydro', 'best_env', 'best_irr']
    irr_index = ['2', '3', '4', '5', '6', '7', '8', '9']
    irr_d = ['Irrigation District 2', 'Irrigation District 3', 'Irrigation District 4', 'Irrigation District 5',
             'Irrigation District 6', 'Irrigation District 7', 'Irrigation District 8', 'Irrigation District 9']
    label_policy = ['Best Hydropower', 'Best Environment', 'Best Irrigation', 'Target Demand']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    n_months = 12
    n_years = 20
    purple = '#7a0177';
    yellow = '#fdaa09';
    blue = '#1d91c0'  # green='#41ab5d'
    colors = [purple, yellow, blue]
    variables_names = [r'$q_t$', r'$h_t$', r'$s_t$', r'$s_{t+1}$', r'$r_{t+1}$', r'$r^{delay}_{t+1}$']#, r'$HP_{t+1}$']
    variables = ['q', 'h_t', 's_t', 's_t+1', 'r_t+1', 'r_d_t+1']#, 'HP_t+1']

    bc_irr_deficits = pd.DataFrame()
    bc_irr_def_target = pd.DataFrame()
    df_concat = pd.DataFrame()

    image_format = ['png']
    for im in range(len(image_format)):
        for policy in range(len(policies)):
            if not os.path.exists(output_folder + '/' + feature + '/' + image_format[im] + '/' + policies[policy]):
                os.makedirs(output_folder + '/' + feature + '/' + image_format[im] + '/' + policies[policy])
    # this generates 8 plots one for each irrigation district:
    for ir in range(len(irr_d)):
        irr_plots(input_folder, target_input_folder, output_folder, feature, ir, irr_d, irr_index, policies, months,
                  label_policy, n_months, n_years, colors)
        bc_def_plot, df_def, df_def_target, df = deficit_plots(input_folder, target_input_folder, output_folder, feature, ir, irr_d, irr_index, policies, months,
                                                           label_policy, n_months, n_years, colors)
        bc_irr_deficits = pd.concat([bc_irr_deficits, df_def])
        bc_irr_def_target = pd.concat([bc_irr_def_target, df_def_target])
        df_concat = pd.concat([df_concat, df], ignore_index=True)
    bc_irr_deficits = bc_irr_deficits.reset_index(drop=True)
    bc_irr_def_target = bc_irr_def_target.reset_index(drop=True)
    #df_concat = df_concat.reset_index(drop=True)
    #print('BC IRR DEF TARGET', bc_irr_def_target, "END BC IR DEF TARGET")
    #print("DF", df.head(20), "END DF")
    #print("DF CONCAT", df_concat.head(20), "END DF CONCAT")
    bc_path = '../runs/BC_BC_pseudo_200000nfe_5seed'
    deficits_name = 'bc_irr_deficits.csv'
    bc_irr_deficits.to_csv(os.path.join(bc_path, deficits_name))
    targets_name = 'bc_irr_deficits_target.csv'
    bc_irr_def_target.to_csv(os.path.join(bc_path, targets_name))
    df_concat.to_csv(os.path.join(bc_path,"df_monthly_ir_deficits.csv"))
    # this a summary of the delta releases:
    mef_plots(input_folder, output_folder, label_policy, delta_release_balance, feature, policies, n_years, n_months,
              delta_target, colors, months, title, target_input_folder)

    #v = 4  # to print only releases across all reservoirs:
    for v in range(len(variables)-1): #to print all summary figures:
        for p in range(len(policies)):
            fig = plt.figure()
            for r in range(len(reservoirs)):
                summary_plot(v, p, r, fig, input_folder, output_folder, feature, policies, variables, label_policy,
                             reservoirs, res_names, months, n_years, n_months)


def irr_plot_quantities():
    plt.rcParams["font.family"] = "Myriad Pro"
    sns.set_style("whitegrid")

    input_folder = '../storage_release/'
    # input_folder_objs='../for_plots/'
    target_input_folder = '../data/'
    output_folder = '../plots/'
    delta_target = np.loadtxt(target_input_folder + 'MEF_delta.txt')

    # copy here..
    #####################################
    feature = 'irr_policy_simulation'
    # reservoirs=['itt','kgu','kgl','ka','bg','dg','cb','mn']
    reservoirs = ['itt', 'kgu', 'kgl', 'ka', 'cb']
    title = '5_res_wKGL'
    # input_file='Zambezi_'+title+'.reference'  #'.reference'change file_name

    # data= np.loadtxt('../parallel/sets/'+feature+'/'+input_file, skiprows=0+1+2-1)
    delta_release_balance = '\n('r'$r_{CB}+Q_{Shire}-r_{Irrd7}-r_{Irrd8}-r_{Irrd9}$)'
    # res_names=['Itezhitezhi','Kafue G. Upper','Kafue G. Lower','Kariba','Batoka Gorge','Devil\'s Gorge','Cahora Bassa', 'Mphanda Nkuwa']
    res_names = ['Itezhitezhi', 'Kafue G. Upper', 'Kafue G. Lower', 'Kariba', 'Cahora Bassa']
    #####copy the segment above#########

    policies = ['best_hydro', 'best_env', 'best_irr','best_irr2', 'best_irr3', 'best_irr4', 'best_irr5', 'best_irr6', 'best_irr7', 'best_irr8', 'best_irr9']
    irr_index = ['2', '3', '4', '5', '6', '7', '8', '9'] #*******************#
    irr_d = ['Irrigation District 2', 'Irrigation District 3', 'Irrigation District 4', 'Irrigation District 5',
             'Irrigation District 6', 'Irrigation District 7', 'Irrigation District 8', 'Irrigation District 9']
    label_policy = ['Best Hydropower', 'Best Environment', 'Best Irrigation', 'Best Irrigation 2', 'Best Irrigation 3', 'Best Irrigation 4', 'Best Irrigation 5', 'Best Irrigation 6', 'Best Irrigation 7', 'Best Irrigation 8', 'Best Irrigation 9', 'Target Demand']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    n_months = 12
    n_years = 20
    '''
    purple = '#7a0177';
    yellow = '#fdaa09';
    blue = '#1d91c0'  # green='#41ab5d'
    colors = [purple, yellow, blue]
    '''
    purple = '#7a0177'
    yellow = '#fdaa09'
    blue = '#1d91c0'
    green = '#41ab5d'
    red = '#e41a1c'
    orange = '#ff7f00'
    cyan = '#a65628'
    magenta = '#f781bf'
    dark_yellow = '#ffba00'
    dark_green = '#4daf4a'
    dark_blue = '#377eb8'
    dark_purple = '#984ea3'

    colors = [blue, green, purple, red, orange, cyan, magenta, dark_yellow, dark_green, dark_blue, dark_purple]

    variables_names = [r'$q_t$', r'$h_t$', r'$s_t$', r'$s_{t+1}$', r'$r_{t+1}$', r'$r^{delay}_{t+1}$']
    variables = ['q', 'h_t', 's_t', 's_t+1', 'r_t+1', 'r_d_t+1']
    image_format = ['png']

    irr_deficits = pd.DataFrame()
    bc_irr_def_target = pd.DataFrame()
    df_concat = pd.DataFrame()

    for im in range(len(image_format)):
        for policy in range(len(policies)):
            if not os.path.exists(output_folder + '/' + feature + '/' + image_format[im] + '/' + policies[policy]):
                os.makedirs(output_folder + '/' + feature + '/' + image_format[im] + '/' + policies[policy])
    # this generates 8 plots one for each irrigation district:
    for ir in range(len(irr_d)):
        irr_plots(input_folder, target_input_folder, output_folder, feature, ir, irr_d, irr_index, policies, months,
                  label_policy, n_months, n_years, colors)

        ir_def_plot, df_def, df_def_target, df = deficit_plots(input_folder, target_input_folder, output_folder, feature, ir, irr_d,
                                                               irr_index, policies, months, label_policy, n_months, n_years, colors)
        irr_deficits = pd.concat([irr_deficits, df_def])
        bc_irr_def_target = pd.concat([bc_irr_def_target, df_def_target])
        df_concat = pd.concat([df_concat, df])
    irr_deficits = irr_deficits.reset_index(drop=True)
    bc_irr_def_target = bc_irr_def_target.reset_index(drop=True)
    df_concat = df_concat.reset_index(drop=True)

    path = '../runs/IR_new_pseudo_mln_1000000nfe_5seed'
    deficits_name = 'irr_irr_deficits.csv'
    irr_deficits.to_csv(os.path.join(path, deficits_name))
    targets_name = 'ir_irr_deficits_target.csv'
    bc_irr_def_target.to_csv(os.path.join(path, targets_name))
    df_concat.to_csv(os.path.join(path, "df_monthly_ir_deficits.csv"))
    # this a summary of the delta releases:
    mef_plots(input_folder, output_folder, label_policy, delta_release_balance, feature, policies, n_years, n_months,
              delta_target, colors, months, title, target_input_folder)

    v = 4  # to print only releases across all reservoirs:
    # for v in range(len(variables)-1): to print all summary figures:
    for p in range(len(policies)):
        fig = plt.figure()
        for r in range(len(reservoirs)):
            summary_plot(v, p, r, fig, input_folder, output_folder, feature, policies, variables, label_policy,
                         reservoirs, res_names, months, n_years, n_months)


def hyd_plot_quantities():
    plt.rcParams["font.family"] = "Myriad Pro"
    sns.set_style("whitegrid")

    input_folder = '../storage_release/'
    # input_folder_objs='../for_plots/'
    target_input_folder = '../data/'
    output_folder = '../plots/'
    delta_target = np.loadtxt(target_input_folder + 'MEF_delta.txt')

    # copy here..
    #####################################
    feature = 'hyd_policy_simulation'
    # reservoirs=['itt','kgu','kgl','ka','bg','dg','cb','mn']
    reservoirs = ['itt', 'kgu', 'kgl', 'ka', 'cb']
    title = '5_res_wKGL'
    # input_file='Zambezi_'+title+'.reference'  #'.reference'change file_name

    # data= np.loadtxt('../parallel/sets/'+feature+'/'+input_file, skiprows=0+1+2-1)
    delta_release_balance = '\n('r'$r_{CB}+Q_{Shire}-r_{Irrd7}-r_{Irrd8}-r_{Irrd9}$)'
    # res_names=['Itezhitezhi','Kafue G. Upper','Kafue G. Lower','Kariba','Batoka Gorge','Devil\'s Gorge','Cahora Bassa', 'Mphanda Nkuwa']
    res_names = ['Itezhitezhi', 'Kafue G. Upper', 'Kafue G. Lower', 'Kariba', 'Cahora Bassa']
    #####copy the segment above#########

    policies = ['best_hydro', 'best_env', 'best_irr',"best_hydITT", "best_hydKGU", "best_hydKA","best_hydCB",'best_hydKGL']
    irr_index = ['2', '3', '4', '5', '6', '7', '8', '9'] #*******************#
    irr_d = ['Irrigation District 2', 'Irrigation District 3', 'Irrigation District 4', 'Irrigation District 5',
             'Irrigation District 6', 'Irrigation District 7', 'Irrigation District 8', 'Irrigation District 9']
    label_policy = ['Best Hydropower', 'Best Environment', 'Best Irrigation', 'Best Hydropower ITT', 'Best Hydropower KGU', 'Best Hydropower KA', 'Best Hydropower CB', 'Best Hydropower KGL', 'Target Demand']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    n_months = 12
    n_years = 20

    # Define colormap
    colormap = plt.cm.get_cmap('tab20')

    # Assign colors from the colormap to label policies
    colors = [colormap(i) for i in np.linspace(0, 1, len(label_policy))]

    variables_names = [r'$q_t$', r'$h_t$', r'$s_t$', r'$s_{t+1}$', r'$r_{t+1}$', r'$r^{delay}_{t+1}$']
    variables = ['q', 'h_t', 's_t', 's_t+1', 'r_t+1', 'r_d_t+1']
    irr_deficits = pd.DataFrame()
    bc_irr_def_target = pd.DataFrame()
    df_concat = pd.DataFrame()

    image_format = ['png']
    for im in range(len(image_format)):
        for policy in range(len(policies)):
            if not os.path.exists(output_folder + '/' + feature + '/' + image_format[im] + '/' + policies[policy]):
                os.makedirs(output_folder + '/' + feature + '/' + image_format[im] + '/' + policies[policy])
    # this generates 8 plots one for each irrigation district:
    for ir in range(len(irr_d)):
        irr_plots(input_folder, target_input_folder, output_folder, feature, ir, irr_d, irr_index, policies, months,
                  label_policy, n_months, n_years, colors)
        ir_def_plot, df_def, df_def_target, df = deficit_plots(input_folder, target_input_folder, output_folder, feature, ir, irr_d,
                                            irr_index, policies, months,
                                            label_policy, n_months, n_years, colors)
        irr_deficits = pd.concat([irr_deficits, df_def])
        bc_irr_def_target = pd.concat([bc_irr_def_target, df_def_target])
        df_concat = pd.concat([df_concat, df])

    df_concat = df_concat.reset_index(drop=True)
    irr_deficits = irr_deficits.reset_index(drop=True)
    bc_irr_def_target = bc_irr_def_target.reset_index(drop=True)
    path = '../runs/HYD_pseudo_200000nfe_5seed'
    deficits_name = 'hyd_irr_deficits.csv'
    irr_deficits.to_csv(os.path.join(path, deficits_name))
    targets_name = 'hyd_irr_deficits_target.csv'
    bc_irr_def_target.to_csv(os.path.join(path, targets_name))
    df_concat.to_csv(os.path.join(path, "df_monthly_ir_deficits.csv"))
    # this a summary of the delta releases:
    mef_plots(input_folder, output_folder, label_policy, delta_release_balance, feature, policies, n_years, n_months,
              delta_target, colors, months, title, target_input_folder)

    v = 4  # to print only releases across all reservoirs:
    # for v in range(len(variables)-1): to print all summary figures:
    for p in range(len(policies)):
        fig = plt.figure()
        for r in range(len(reservoirs)):
            summary_plot(v, p, r, fig, input_folder, output_folder, feature, policies, variables, label_policy,
                         reservoirs, res_names, months, n_years, n_months)


def full_plot_quantities():
    plt.rcParams["font.family"] = "Myriad Pro"
    sns.set_style("whitegrid")

    input_folder = '../storage_release/'
    # input_folder_objs='../for_plots/'
    target_input_folder = '../data/'
    output_folder = '../plots/'
    delta_target = np.loadtxt(target_input_folder + 'MEF_delta.txt')

    # copy here..
    #####################################
    feature = 'full_policy_simulation'
    # reservoirs=['itt','kgu','kgl','ka','bg','dg','cb','mn']
    reservoirs = ['itt', 'kgu', 'kgl', 'ka', 'cb']
    title = '5_res_wKGL'
    # input_file='Zambezi_'+title+'.reference'  #'.reference'change file_name

    # data= np.loadtxt('../parallel/sets/'+feature+'/'+input_file, skiprows=0+1+2-1)
    delta_release_balance = '\n('r'$r_{CB}+Q_{Shire}-r_{Irrd7}-r_{Irrd8}-r_{Irrd9}$)'
    # res_names=['Itezhitezhi','Kafue G. Upper','Kafue G. Lower','Kariba','Batoka Gorge','Devil\'s Gorge','Cahora Bassa', 'Mphanda Nkuwa']
    res_names = ['Itezhitezhi', 'Kafue G. Upper', 'Kafue G. Lower', 'Kariba', 'Cahora Bassa']
    #####copy the segment above#########

    policies = ['best_hydro', 'best_env', 'best_irr', 'best_irr2', 'best_irr3', 'best_irr4', 'best_irr5', 'best_irr6', 'best_irr7', 'best_irr8', 'best_irr9',"best_hydITT", "best_hydKGU", "best_hydKA","best_hydCB",'best_hydKGL']
    irr_index = ['2', '3', '4', '5', '6', '7', '8', '9'] #*******************#
    irr_d = ['Irrigation District 2', 'Irrigation District 3', 'Irrigation District 4', 'Irrigation District 5',
             'Irrigation District 6', 'Irrigation District 7', 'Irrigation District 8', 'Irrigation District 9']
    label_policy = ['Best Hydropower', 'Best Environment', 'Best Irrigation', 'Best Irrigation 2', 'Best Irrigation 3', 'Best Irrigation 4', 'Best Irrigation 5', 'Best Irrigation 6', 'Best Irrigation 7', 'Best Irrigation 8', 'Best Irrigation 9', 'Best Hydropower ITT', 'Best Hydropower KGU', 'Best Hydropower KA', 'Best Hydropower CB', 'Best Hydropower KGL', 'Target Demand']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    n_months = 12
    n_years = 20

    # Define colormap
    colormap = plt.cm.get_cmap('tab20')

    # Assign colors from the colormap to label policies
    colors = [colormap(i) for i in np.linspace(0, 1, len(label_policy))]

    variables_names = [r'$q_t$', r'$h_t$', r'$s_t$', r'$s_{t+1}$', r'$r_{t+1}$', r'$r^{delay}_{t+1}$']
    variables = ['q', 'h_t', 's_t', 's_t+1', 'r_t+1', 'r_d_t+1']
    irr_deficits = pd.DataFrame()
    bc_irr_def_target = pd.DataFrame()
    df_concat = pd.DataFrame()
    image_format = ['png']
    for im in range(len(image_format)):
        for policy in range(len(policies)):
            if not os.path.exists(output_folder + '/' + feature + '/' + image_format[im] + '/' + policies[policy]):
                os.makedirs(output_folder + '/' + feature + '/' + image_format[im] + '/' + policies[policy])
    # this generates 8 plots one for each irrigation district:
    for ir in range(len(irr_d)):
        irr_plots(input_folder, target_input_folder, output_folder, feature, ir, irr_d, irr_index, policies, months,
                  label_policy, n_months, n_years, colors)
        ir_def_plot, df_def,df_def_target, df = deficit_plots(input_folder, target_input_folder, output_folder, feature, ir, irr_d,
                                            irr_index, policies, months,
                                            label_policy, n_months, n_years, colors)
        irr_deficits = pd.concat([irr_deficits, df_def])
        bc_irr_def_target = pd.concat([bc_irr_def_target, df_def_target])
        df_concat = pd.concat([df_concat, df])
    df_concat = df_concat.reset_index(drop=True)
    irr_deficits = irr_deficits.reset_index(drop=True)
    print(irr_deficits)
    bc_irr_def_target = bc_irr_def_target.reset_index(drop=True)
    print(bc_irr_def_target)
    path = '../runs/FULL_pseudo_mln_1000000nfe_5seed'
    deficits_name = 'full_irr_deficits.csv'
    irr_deficits.to_csv(os.path.join(path, deficits_name))
    targets_name = 'full_irr_deficits_target.csv'
    bc_irr_def_target.to_csv(os.path.join(path, targets_name))
    df_concat.to_csv(os.path.join(path, "df_monthly_ir_deficits.csv"))

    # this a summary of the delta releases:
    mef_plots(input_folder, output_folder, label_policy, delta_release_balance, feature, policies, n_years, n_months,
              delta_target, colors, months, title, target_input_folder)

    v = 4  # to print only releases across all reservoirs:
    # for v in range(len(variables)-1): to print all summary figures:
    for p in range(len(policies)):
        fig = plt.figure()
        for r in range(len(reservoirs)):
            summary_plot(v, p, r, fig, input_folder, output_folder, feature, policies, variables, label_policy,
                         reservoirs, res_names, months, n_years, n_months)


def irr_plots(input_folder, t_irr_folder, output_folder, feature, ir, irr_d, irr_index, policies, months, label_policy,
              n_months, n_years, colors):
    left = 0.05;
    bottom = 0.17;
    right = 0.98;
    top = 0.89;
    wspace = 0.2;
    hspace = 0.2
    font_size = 22
    font_sizey = 22
    font_size_title = 25

    # for ir in range(len(irr_d)):
    fig = plt.figure()
    for p in range(len(policies)):
        # actual release for irrigation:
        data = np.loadtxt(input_folder + feature + '/irr_' + policies[p] + '.txt')
        # irrigation target demand:
        data2 = np.loadtxt(t_irr_folder + 'IrrDemand' + irr_index[ir] + '.txt')
        irrigation = np.reshape(data[:, ir], (n_years, n_months))

        mean_irr = np.mean(irrigation, 0)
        min_irr = np.min(irrigation, 0)
        max_irr = np.max(irrigation, 0)

        if label_policy[p] in ['Best Hydropower', 'Best Irrigation', 'Best Environment']:
            line_width = 4  # Thicker lines for specific labels
        else:
            line_width = 2  # Default line width for other labels

        plt.fill_between(range(n_months), max_irr, min_irr, alpha=0.5, color=colors[p])
        plt.plot(mean_irr, linewidth=line_width, color=colors[p], label=label_policy[p])

        plt.title(irr_d[ir], fontsize=font_size_title)
        plt.ylabel('Average diversion bounded \nby min and max values [m'r'$^3$/sec]', fontsize=font_sizey, labelpad=20)
        plt.xticks(np.arange(n_months), months, rotation=30, fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xlim([0, 11])

    plt.plot(data2, color='k', linestyle=':', linewidth=5, label='Target Demand')
    plt.legend(loc='upper left', fontsize=font_size)
    fig.set_size_inches(12, 10)
    return plt.savefig('../plots/' + feature + '/png/irr_d_' + irr_index[ir] + '.png')


def deficit_plots(input_folder, t_irr_folder, output_folder, feature, ir, irr_d, irr_index, policies, months, label_policy,
              n_months, n_years, colors):
    left = 0.05;
    bottom = 0.17;
    right = 0.98;
    top = 0.89;
    wspace = 0.2;
    hspace = 0.2
    font_size = 22
    font_sizey = 22
    font_size_title = 25
    df_def_monthly = pd.DataFrame()
    df_def_target = pd.DataFrame()
    df_def = pd.DataFrame()
    irr_def_list = []
    # for ir in range(len(irr_d)):
    fig = plt.figure()
    for p in range(3):

        # actual release for irrigation:
        data = np.loadtxt(input_folder + feature + '/irr_' + policies[p] + '.txt')
        # irrigation target demand:
        irr_target_demand = np.loadtxt(t_irr_folder + 'IrrDemand' + irr_index[ir] + '.txt')
        irrigation = np.reshape(data[:, ir], (n_years, n_months))

        mean_irr = np.mean(irrigation, 0)
        min_irr = np.min(irrigation, 0)
        max_irr = np.max(irrigation, 0)
        #############

        deficit = np.empty(0)
        deficit_sq = np.empty(0)
        deficit_sq_norm = np.empty(0)
        for moy in range(12):
            #print(moy)
            ir_deficit = max(irr_target_demand[moy] - mean_irr[moy], 0)
            deficit = np.append(deficit, ir_deficit) # the unedited deficit is used for the visualization and comparison
            deficit_temp = pow(max(irr_target_demand[moy] - mean_irr[moy], 0), 2) #Squared deficit
            deficit_sq = np.append(deficit_sq, deficit_temp)
            deficit_sq_norm_temp = g_deficit_norm(deficit_temp, irr_target_demand[moy])
            deficit_sq_norm = np.append(deficit_sq_norm, deficit_sq_norm_temp)


        if label_policy[p] in ['Best Hydropower', 'Best Irrigation', 'Best Environment']:
            line_width = 4  # Thicker lines for specific labels
        else:
            line_width = 2  # Default line width for other labels

        plt.plot(deficit, linewidth=line_width, color=colors[p], label=label_policy[p])
        plt.title(irr_d[ir], fontsize=font_size_title)
        plt.ylabel('Irrigation deficit \n [m'r'$^3$/sec]', fontsize=font_sizey, labelpad=20)
        plt.xticks(np.arange(n_months), months, rotation=30, fontsize=font_size)
        plt.yticks(fontsize=font_size)
        #irr_def_list.append(deficit)
        irr_def_list.append([irr_d[ir], label_policy[p]] + deficit.tolist())

        #print(ir, label_policy[p], deficit)

        df_pol = pd.DataFrame({
            'District': [int(ir)+2],
            'Policy': label_policy[p],
            'Deficit': [np.mean(deficit)]
        })
        #print(df_pol)
        df_def = pd.concat([df_def, df_pol])

        df_pol_target = pd.DataFrame({
            'District': [int(ir)+2],
            'Policy': label_policy[p],
            'Deficit': [np.mean(deficit)],
            'Target': [np.mean(irr_target_demand)]
        })
        # Calculate the relative deficit
        df_pol_target['Relative Deficit'] = np.divide(df_pol_target['Deficit'], df_pol_target['Target'])

        df_def_target = pd.concat([df_def_target, df_pol_target])
   # df = pd.DataFrame(irr_def_list, columns=['district', 'policy', 'irrigation_deficit'])

    plt.legend(loc='upper left', fontsize=font_size)
    fig.set_size_inches(12, 10)

    month_columns = [f'Month_{i + 1}' for i in range(n_months)]
    df = pd.DataFrame(irr_def_list, columns=['district', 'policy'] + month_columns)

    #print('Deficit district:',ir,deficit)
    #print(df_def)

    return plt.savefig('../plots/' + feature + '/png/irr_def_3obj' + irr_index[ir] + '.png'), df_def, df_def_target, df #change name


def g_deficit_norm(defp, w):
    """Takes two floats and divides the first by the square of the second.

    Parameters
    ----------
    defp : float
    w : float

    Returns
    -------
    def_norm : float
    """

    def_norm = 0
    if (w == 0.0):
        def_norm = 0.0
    else:
        def_norm = defp / (pow(w, 2))

    return def_norm


def mef_plots(input_folder, output_folder, label_policy, delta_release_balance, feature, policies, n_years, n_months,
              delta_target, colors, months, title, mef_folder):
    left = 0.18;
    bottom = 0.1;
    right = 0.96;
    top = 0.92;
    wspace = 0.2;
    hspace = 0.2
    font_size = 22
    font_sizey = 22
    font_size_title = 25
    fig = plt.figure()
    for p in range(len(policies)):
        # actual release for mef_
        data = np.loadtxt(input_folder + feature + '/rDelta_' + policies[p] + '.txt')
        # target mef
        rMEF = np.reshape(data, (n_years, n_months))
        mean_mef = np.mean(rMEF, 0)
        min_mef = np.min(rMEF, 0)
        max_mef = np.max(rMEF, 0)

        plt.fill_between(range(n_months), max_mef, min_mef, alpha=0.5, color=colors[p])
        plt.plot(mean_mef, linewidth=5, color=colors[p], label=label_policy[p])

        plt.title('Delta releases-' + title + delta_release_balance, fontsize=font_size_title)
        plt.ylabel('Average environmental flows bounded\nby min and max values [m'r'$^3$/sec]', fontsize=font_sizey,
                   labelpad=20)
        plt.xticks(np.arange(n_months), months, rotation=30, fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.xlim([0, 11])

    plt.plot(delta_target, color='k', linestyle=':', linewidth=6, label='MEF Delta target')
    plt.legend(fontsize=font_size)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    fig.set_size_inches(12, 10)
    return plt.savefig(output_folder + feature + '/png/rMEF.png')


def summary_plot(v, p, r, fig, input_folder, output_folder, feature, policies, variables, label_policy, reservoirs,
                 res_names, months, n_years, n_months):
    #colorsr = ['#b2182b', '#d6604d', '#fc8d59', '#f4a582', '#92c5de', '#6baed6', '#4393c3', '#2166ac']
    colorsr = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0']

    left = 0.13;
    bottom = 0.12;
    right = 0.75;
    top = 0.95;
    wspace = 0.2;
    hspace = 0.2
    font_size = 18
    font_sizey = 20
    font_size_title = 25
    y_label = ['Inflow [m'r'$^3$/sec]', 'Level (t) [m]', 'Storage (t) [m'r'$^3$]', 'Storage (t+1) [m'r'$^3$]',
               'Average Release (t+1) [m'r'$^3$/sec]', 'Average Release (t+2) [m'r'$^3$/sec]', 'Hydropower deficit (t) [TWh/month]']
    locs, labels = plt.xticks()
    data = np.loadtxt(input_folder + feature + '/' + reservoirs[r] + '_' + policies[p] + '.txt')
    data = np.reshape(data[:, v], (n_years, n_months))
    avg = np.mean(data, 0)
    plt.plot(avg, color=colorsr[r], linewidth=7, linestyle=':', label=res_names[r])
    plt.xticks(np.arange(n_months), months, rotation=30, fontsize=font_size)
    plt.ylabel(y_label[v], fontsize=font_size_title, labelpad=30)
    plt.yticks(fontsize=font_sizey)
    plt.title(label_policy[p], fontsize=font_size_title)
    plt.xlim([0, 11])
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    fig.set_size_inches(14, 10)
    plt.legend(fontsize=font_sizey, labelspacing=3, loc=6, bbox_to_anchor=(
    1, 0.5))  # bbox_to_anchor=(0., 1.02, 1., .102), loc=3,fontsize=font_sizey, ncol=4, mode="expand")

    return plt.savefig(output_folder + feature + '/png/' + variables[v] + '_all_reservoirs_' + policies[p] + '.png')

