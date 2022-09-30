import pandas as pd
from dotenv import dotenv_values

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

config = dotenv_values('../.env')
train = pd.read_parquet(config["ENGINEERED_DATA"] + "train_fe.parquet")

train_id = train["customer_ID"]
train_target = train["target"]
FEATURES = train.columns[1:-1]

categorical = [
            'D_44_std', 'R_2_std', 'D_51_std', 'S_6_std', 'D_54_max', 'R_4_std',
            'R_5_std', 'B_22_std', 'D_70_std', 'D_72_std', 'D_73_mean',
            'D_73_std', 'D_73_min', 'D_73_max', 'D_73_last', 'D_76_mean',
            'D_76_std', 'D_76_min', 'D_76_max', 'D_76_last', 'D_78_std',
            'D_79_std', 'R_8_std', 'R_9_std', 'D_80_std', 'R_10_std',
            'R_11_std', 'D_81_std', 'D_82_std', 'R_13_std', 'D_83_std',
            'R_15_std', 'D_84_std', 'R_16_std', 'B_29_mean', 'B_29_std',
            'B_29_min', 'B_29_max', 'B_29_last', 'S_18_std', 'D_86_std',
            'D_87_std', 'R_17_std', 'R_18_std', 'D_88_mean', 'D_88_std',
            'D_88_min', 'D_88_max', 'D_88_last', 'B_31_std', 'R_19_std',
            'B_32_std', 'S_20_std', 'R_20_std', 'R_21_std', 'B_33_std',
            'D_89_std', 'R_22_std', 'R_23_std', 'D_91_std', 'D_92_std',
            'D_93_std', 'D_94_std', 'R_24_std', 'R_25_std', 'D_96_std',
            'D_103_std', 'D_107_std', 'R_26_std', 'D_108_std', 'D_109_std',
            'D_110_mean', 'D_110_std', 'D_110_min', 'D_110_max', 'D_110_last',
            'D_111_std', 'B_39_mean', 'B_39_std', 'B_39_min', 'B_39_max',
            'B_39_last', 'D_113_std', 'D_122_std', 'D_123_std', 'D_124_std',
            'D_125_std', 'D_127_std', 'D_129_std', 'B_41_std', 'B_42_mean',
            'B_42_std', 'B_42_min', 'B_42_max', 'B_42_last', 'D_132_mean',
            'D_132_std', 'D_132_min', 'D_132_max', 'D_132_last', 'R_28_std',
            'D_134_mean', 'D_134_std', 'D_134_min', 'D_134_max', 'D_134_last',
            'D_135_std', 'D_136_std', 'D_137_std', 'D_138_std', 'D_139_std',
            'D_140_std', 'D_143_std', 'D_145_std'
        ]

numeric = [
            'P_2_mean', 'P_2_std', 'P_2_min', 'P_2_max', 'P_2_last',
            'D_39_mean', 'D_39_std', 'D_39_min', 'D_39_max', 'D_39_last',
            'B_1_mean', 'B_1_std', 'B_1_min', 'B_1_max', 'B_1_last', 'B_2_mean',
            'B_2_std', 'B_2_min', 'B_2_max', 'B_2_last', 'R_1_mean', 'R_1_std',
            'R_1_min', 'R_1_max', 'R_1_last', 'S_3_mean', 'S_3_std', 'S_3_min',
            'S_3_max', 'S_3_last', 'D_41_mean', 'D_41_std', 'D_41_min',
            'D_41_max', 'D_41_last', 'B_3_mean', 'B_3_std', 'B_3_min',
            'B_3_max', 'B_3_last', 'D_42_mean', 'D_42_std', 'D_42_min',
            'D_42_max', 'D_42_last', 'D_43_mean', 'D_43_std', 'D_43_min',
            'D_43_max', 'D_43_last', 'D_44_mean', 'D_44_min', 'D_44_max',
            'D_44_last', 'B_4_mean', 'B_4_std', 'B_4_min', 'B_4_max',
            'B_4_last', 'D_45_mean', 'D_45_std', 'D_45_min', 'D_45_max',
            'D_45_last', 'B_5_mean', 'B_5_std', 'B_5_min', 'B_5_max',
            'B_5_last', 'R_2_mean', 'R_2_min', 'R_2_max', 'R_2_last',
            'D_46_mean', 'D_46_std', 'D_46_min', 'D_46_max', 'D_46_last',
            'D_47_mean', 'D_47_std', 'D_47_min', 'D_47_max', 'D_47_last',
            'D_48_mean', 'D_48_std', 'D_48_min', 'D_48_max', 'D_48_last',
            'D_49_mean', 'D_49_std', 'D_49_min', 'D_49_max', 'D_49_last',
            'B_6_mean', 'B_6_std', 'B_6_min', 'B_6_max', 'B_6_last', 'B_7_mean',
            'B_7_std', 'B_7_min', 'B_7_max', 'B_7_last', 'B_8_mean', 'B_8_std',
            'B_8_min', 'B_8_max', 'B_8_last', 'D_50_mean', 'D_50_std',
            'D_50_min', 'D_50_max', 'D_50_last', 'D_51_mean', 'D_51_min',
            'D_51_max', 'D_51_last', 'B_9_mean', 'B_9_std', 'B_9_min',
            'B_9_max', 'B_9_last', 'R_3_mean', 'R_3_std', 'R_3_min', 'R_3_max',
            'R_3_last', 'D_52_mean', 'D_52_std', 'D_52_min', 'D_52_max',
            'D_52_last', 'P_3_mean', 'P_3_std', 'P_3_min', 'P_3_max',
            'P_3_last', 'B_10_mean', 'B_10_std', 'B_10_min', 'B_10_max',
            'B_10_last', 'D_53_mean', 'D_53_std', 'D_53_min', 'D_53_max',
            'D_53_last', 'S_5_mean', 'S_5_std', 'S_5_min', 'S_5_max',
            'S_5_last', 'B_11_mean', 'B_11_std', 'B_11_min', 'B_11_max',
            'B_11_last', 'S_6_mean', 'S_6_min', 'S_6_max', 'S_6_last',
            'D_54_mean', 'D_54_std', 'D_54_min', 'D_54_last', 'R_4_mean',
            'R_4_min', 'R_4_max', 'R_4_last', 'S_7_mean', 'S_7_std', 'S_7_min',
            'S_7_max', 'S_7_last', 'B_12_mean', 'B_12_std', 'B_12_min',
            'B_12_max', 'B_12_last', 'S_8_mean', 'S_8_std', 'S_8_min',
            'S_8_max', 'S_8_last', 'D_55_mean', 'D_55_std', 'D_55_min',
            'D_55_max', 'D_55_last', 'D_56_mean', 'D_56_std', 'D_56_min',
            'D_56_max', 'D_56_last', 'B_13_mean', 'B_13_std', 'B_13_min',
            'B_13_max', 'B_13_last', 'R_5_mean', 'R_5_min', 'R_5_max',
            'R_5_last', 'D_58_mean', 'D_58_std', 'D_58_min', 'D_58_max',
            'D_58_last', 'S_9_mean', 'S_9_std', 'S_9_min', 'S_9_max',
            'S_9_last', 'B_14_mean', 'B_14_std', 'B_14_min', 'B_14_max',
            'B_14_last', 'D_59_mean', 'D_59_std', 'D_59_min', 'D_59_max',
            'D_59_last', 'D_60_mean', 'D_60_std', 'D_60_min', 'D_60_max',
            'D_60_last', 'D_61_mean', 'D_61_std', 'D_61_min', 'D_61_max',
            'D_61_last', 'B_15_mean', 'B_15_std', 'B_15_min', 'B_15_max',
            'B_15_last', 'S_11_mean', 'S_11_std', 'S_11_min', 'S_11_max',
            'S_11_last', 'D_62_mean', 'D_62_std', 'D_62_min', 'D_62_max',
            'D_62_last', 'D_65_mean', 'D_65_std', 'D_65_min', 'D_65_max',
            'D_65_last', 'B_16_mean', 'B_16_std', 'B_16_min', 'B_16_max',
            'B_16_last', 'B_17_mean', 'B_17_std', 'B_17_min', 'B_17_max',
            'B_17_last', 'B_18_mean', 'B_18_std', 'B_18_min', 'B_18_max',
            'B_18_last', 'B_19_mean', 'B_19_std', 'B_19_min', 'B_19_max',
            'B_19_last', 'B_20_mean', 'B_20_std', 'B_20_min', 'B_20_max',
            'B_20_last', 'S_12_mean', 'S_12_std', 'S_12_min', 'S_12_max',
            'S_12_last', 'R_6_mean', 'R_6_std', 'R_6_min', 'R_6_max',
            'R_6_last', 'S_13_mean', 'S_13_std', 'S_13_min', 'S_13_max',
            'S_13_last', 'B_21_mean', 'B_21_std', 'B_21_min', 'B_21_max',
            'B_21_last', 'D_69_mean', 'D_69_std', 'D_69_min', 'D_69_max',
            'D_69_last', 'B_22_mean', 'B_22_min', 'B_22_max', 'B_22_last',
            'D_70_mean', 'D_70_min', 'D_70_max', 'D_70_last', 'D_71_mean',
            'D_71_std', 'D_71_min', 'D_71_max', 'D_71_last', 'D_72_mean',
            'D_72_min', 'D_72_max', 'D_72_last', 'S_15_mean', 'S_15_std',
            'S_15_min', 'S_15_max', 'S_15_last', 'B_23_mean', 'B_23_std',
            'B_23_min', 'B_23_max', 'B_23_last', 'P_4_mean', 'P_4_std',
            'P_4_min', 'P_4_max', 'P_4_last', 'D_74_mean', 'D_74_std',
            'D_74_min', 'D_74_max', 'D_74_last', 'D_75_mean', 'D_75_std',
            'D_75_min', 'D_75_max', 'D_75_last', 'B_24_mean', 'B_24_std',
            'B_24_min', 'B_24_max', 'B_24_last', 'R_7_mean', 'R_7_std',
            'R_7_min', 'R_7_max', 'R_7_last', 'D_77_mean', 'D_77_std',
            'D_77_min', 'D_77_max', 'D_77_last', 'B_25_mean', 'B_25_std',
            'B_25_min', 'B_25_max', 'B_25_last', 'B_26_mean', 'B_26_std',
            'B_26_min', 'B_26_max', 'B_26_last', 'D_78_mean', 'D_78_min',
            'D_78_max', 'D_78_last', 'D_79_mean', 'D_79_min', 'D_79_max',
            'D_79_last', 'R_8_mean', 'R_8_min', 'R_8_max', 'R_8_last',
            'R_9_mean', 'R_9_min', 'R_9_max', 'R_9_last', 'S_16_mean',
            'S_16_std', 'S_16_min', 'S_16_max', 'S_16_last', 'D_80_mean',
            'D_80_min', 'D_80_max', 'D_80_last', 'R_10_mean', 'R_10_min',
            'R_10_max', 'R_10_last', 'R_11_mean', 'R_11_min', 'R_11_max',
            'R_11_last', 'B_27_mean', 'B_27_std', 'B_27_min', 'B_27_max',
            'B_27_last', 'D_81_mean', 'D_81_min', 'D_81_max', 'D_81_last',
            'D_82_mean', 'D_82_min', 'D_82_max', 'D_82_last', 'S_17_mean',
            'S_17_std', 'S_17_min', 'S_17_max', 'S_17_last', 'R_12_mean',
            'R_12_std', 'R_12_min', 'R_12_max', 'R_12_last', 'B_28_mean',
            'B_28_std', 'B_28_min', 'B_28_max', 'B_28_last', 'R_13_mean',
            'R_13_min', 'R_13_max', 'R_13_last', 'D_83_mean', 'D_83_min',
            'D_83_max', 'D_83_last', 'R_14_mean', 'R_14_std', 'R_14_min',
            'R_14_max', 'R_14_last', 'R_15_mean', 'R_15_min', 'R_15_max',
            'R_15_last', 'D_84_mean', 'D_84_min', 'D_84_max', 'D_84_last',
            'R_16_mean', 'R_16_min', 'R_16_max', 'R_16_last', 'S_18_mean',
            'S_18_min', 'S_18_max', 'S_18_last', 'D_86_mean', 'D_86_min',
            'D_86_max', 'D_86_last', 'D_87_mean', 'D_87_min', 'D_87_max',
            'D_87_last', 'R_17_mean', 'R_17_min', 'R_17_max', 'R_17_last',
            'R_18_mean', 'R_18_min', 'R_18_max', 'R_18_last', 'B_31_mean',
            'B_31_min', 'B_31_max', 'B_31_last', 'S_19_mean', 'S_19_std',
            'S_19_min', 'S_19_max', 'S_19_last', 'R_19_mean', 'R_19_min',
            'R_19_max', 'R_19_last', 'B_32_mean', 'B_32_min', 'B_32_max',
            'B_32_last', 'S_20_mean', 'S_20_min', 'S_20_max', 'S_20_last',
            'R_20_mean', 'R_20_min', 'R_20_max', 'R_20_last', 'R_21_mean',
            'R_21_min', 'R_21_max', 'R_21_last', 'B_33_mean', 'B_33_min',
            'B_33_max', 'B_33_last', 'D_89_mean', 'D_89_min', 'D_89_max',
            'D_89_last', 'R_22_mean', 'R_22_min', 'R_22_max', 'R_22_last',
            'R_23_mean', 'R_23_max', 'R_23_last', 'D_91_mean', 'D_91_min',
            'D_91_max', 'D_91_last', 'D_92_mean', 'D_92_min', 'D_92_max',
            'D_92_last', 'D_93_mean', 'D_93_min', 'D_93_max', 'D_93_last',
            'D_94_mean', 'D_94_min', 'D_94_max', 'D_94_last', 'R_24_mean',
            'R_24_min', 'R_24_max', 'R_24_last', 'R_25_mean', 'R_25_min',
            'R_25_max', 'R_25_last', 'D_96_mean', 'D_96_min', 'D_96_max',
            'D_96_last', 'S_22_mean', 'S_22_std', 'S_22_min', 'S_22_max',
            'S_22_last', 'S_23_mean', 'S_23_std', 'S_23_min', 'S_23_max',
            'S_23_last', 'S_24_mean', 'S_24_std', 'S_24_min', 'S_24_max',
            'S_24_last', 'S_25_mean', 'S_25_std', 'S_25_min', 'S_25_max',
            'S_25_last', 'S_26_mean', 'S_26_std', 'S_26_min', 'S_26_max',
            'S_26_last', 'D_102_mean', 'D_102_std', 'D_102_min', 'D_102_max',
            'D_102_last', 'D_103_mean', 'D_103_min', 'D_103_max', 'D_103_last',
            'D_104_mean', 'D_104_std', 'D_104_min', 'D_104_max', 'D_104_last',
            'D_105_mean', 'D_105_std', 'D_105_min', 'D_105_max', 'D_105_last',
            'D_106_mean', 'D_106_std', 'D_106_min', 'D_106_max', 'D_106_last',
            'D_107_mean', 'D_107_min', 'D_107_max', 'D_107_last', 'B_36_mean',
            'B_36_std', 'B_36_min', 'B_36_max', 'B_36_last', 'B_37_mean',
            'B_37_std', 'B_37_min', 'B_37_max', 'B_37_last', 'R_26_mean',
            'R_26_min', 'R_26_max', 'R_26_last', 'R_27_mean', 'R_27_std',
            'R_27_min', 'R_27_max', 'R_27_last', 'D_108_mean', 'D_108_min',
            'D_108_max', 'D_108_last', 'D_109_mean', 'D_109_min', 'D_109_max',
            'D_109_last', 'D_111_mean', 'D_111_min', 'D_111_max', 'D_111_last',
            'D_112_mean', 'D_112_std', 'D_112_min', 'D_112_max', 'D_112_last',
            'B_40_mean', 'B_40_std', 'B_40_min', 'B_40_max', 'B_40_last',
            'S_27_mean', 'S_27_std', 'S_27_min', 'S_27_max', 'S_27_last',
            'D_113_mean', 'D_113_min', 'D_113_max', 'D_113_last', 'D_115_mean',
            'D_115_std', 'D_115_min', 'D_115_max', 'D_115_last', 'D_118_mean',
            'D_118_std', 'D_118_min', 'D_118_max', 'D_118_last', 'D_119_mean',
            'D_119_std', 'D_119_min', 'D_119_max', 'D_119_last', 'D_121_mean',
            'D_121_std', 'D_121_min', 'D_121_max', 'D_121_last', 'D_122_mean',
            'D_122_min', 'D_122_max', 'D_122_last', 'D_123_mean', 'D_123_min',
            'D_123_max', 'D_123_last', 'D_124_mean', 'D_124_min', 'D_124_max',
            'D_124_last', 'D_125_mean', 'D_125_min', 'D_125_max', 'D_125_last',
            'D_127_mean', 'D_127_min', 'D_127_max', 'D_127_last', 'D_128_mean',
            'D_128_std', 'D_128_min', 'D_128_max', 'D_128_last', 'D_129_mean',
            'D_129_min', 'D_129_max', 'D_129_last', 'B_41_mean', 'B_41_min',
            'B_41_max', 'B_41_last', 'D_130_mean', 'D_130_std', 'D_130_min',
            'D_130_max', 'D_130_last', 'D_131_mean', 'D_131_std', 'D_131_min',
            'D_131_max', 'D_131_last', 'D_133_mean', 'D_133_std', 'D_133_min',
            'D_133_max', 'D_133_last', 'R_28_mean', 'R_28_min', 'R_28_max',
            'R_28_last', 'D_135_mean', 'D_135_min', 'D_135_max', 'D_135_last',
            'D_136_mean', 'D_136_min', 'D_136_max', 'D_136_last', 'D_137_mean',
            'D_137_min', 'D_137_max', 'D_137_last', 'D_138_mean', 'D_138_min',
            'D_138_max', 'D_138_last', 'D_139_mean', 'D_139_min', 'D_139_max',
            'D_139_last', 'D_140_mean', 'D_140_min', 'D_140_max', 'D_140_last',
            'D_141_mean', 'D_141_std', 'D_141_min', 'D_141_max', 'D_141_last',
            'D_142_mean', 'D_142_std', 'D_142_min', 'D_142_max', 'D_142_last',
            'D_143_mean', 'D_143_min', 'D_143_max', 'D_143_last', 'D_144_mean',
            'D_144_std', 'D_144_min', 'D_144_max', 'D_144_last', 'D_145_mean',
            'D_145_min', 'D_145_max', 'D_145_last', 'B_30_count', 'B_30_last',
            'B_30_nunique', 'B_38_count', 'B_38_last', 'B_38_nunique',
            'D_114_count', 'D_114_last', 'D_114_nunique', 'D_116_count',
            'D_116_last', 'D_116_nunique', 'D_117_count', 'D_117_last',
            'D_117_nunique', 'D_120_count', 'D_120_last', 'D_120_nunique',
            'D_126_count', 'D_126_last', 'D_126_nunique', 'D_63_count',
            'D_63_last', 'D_63_nunique', 'D_64_count', 'D_64_last',
            'D_64_nunique', 'D_66_count', 'D_66_last', 'D_66_nunique',
            'D_68_count', 'D_68_last', 'D_68_nunique'
        ]

categorical_processor = Pipeline(
    steps=[
        ('catimputer', SimpleImputer(strategy="most_frequent", add_indicator=True))
    ]
)

numerical_processor = Pipeline(
    steps=[
        ('numimputer', SimpleImputer(strategy='median', add_indicator=True))
    ]
)

column_transformer = ColumnTransformer(
    transformers=[
        ('categorical_processing', categorical_processor, categorical),
        ('numeric_processing', numerical_processor, numeric)
    ]
)

pipeline = Pipeline(steps=[('column_transformer', column_transformer)])

imputed_train = pd.DataFrame(pipeline.fit_transform(train[FEATURES]))
imputed_train["customer_ID"] = train_id
imputed_train["target"] = train_target
imputed_train.set_index("customer_ID", inplace=True)
imputed_train.reset_index(inplace=True)
imputed_train.columns = [str(c) for c in imputed_train.columns]
imputed_train.to_csv(config["ENGINEERED_DATA"] + "imputed_train.csv", index=False)

del train, imputed_train
test = pd.read_parquet(config["ENGINEERED_DATA"] + "test_fe.parquet")
test_id = test["customer_ID"]
imputed_test = pd.DataFrame(pipeline.transform(test[FEATURES]))
imputed_test["customer_ID"] = test_id
imputed_test.set_index("customer_ID", inplace=True)
imputed_test.reset_index(inplace=True)
imputed_test.columns = [str(c) for c in imputed_test.columns]
imputed_test.to_csv(config["ENGINEERED_DATA"] + "imputed_test.csv", index=False)
