import joblib
import pandas as pd

with open('model_analysis_output.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("MODEL ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    try:
        model_dict = joblib.load('models/registry/champion_model.joblib')
        f.write(f"Model loaded successfully\n")
        f.write(f"Type: {type(model_dict)}\n\n")
        
        if isinstance(model_dict, dict):
            f.write(f"Dictionary keys: {list(model_dict.keys())}\n\n")
            model = model_dict['model']
            feature_names = model_dict.get('feature_names', None)
            
            f.write(f"Model type: {type(model).__name__}\n")
            f.write(f"Feature names: {len(feature_names) if feature_names else 'None'}\n\n")
            
            if feature_names:
                f.write(f"is_weekend present: {'is_weekend' in feature_names}\n")
                f.write(f"is_festivo present: {'is_festivo' in feature_names}\n\n")
                
                if hasattr(model, 'feature_importances_'):
                    df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    f.write("TOP 20 FEATURES:\n")
                    f.write("-"*80 + "\n")
                    for i, row in df.head(20).iterrows():
                        f.write(f"{row['feature']:40s} {row['importance']:.6f}\n")
                    
                    f.write("\n" + "="*80 + "\n")
                    f.write("WEEKEND/FESTIVO FEATURES:\n")
                    f.write("="*80 + "\n")
                    
                    weekend_row = df[df['feature'] == 'is_weekend']
                    if not weekend_row.empty:
                        rank = df.index.get_loc(weekend_row.index[0]) + 1
                        imp = weekend_row.iloc[0]['importance']
                        f.write(f"\nis_weekend:\n")
                        f.write(f"  Rank: #{rank} / {len(df)}\n")
                        f.write(f"  Importance: {imp:.6f}\n")
                        
                        if rank > 30:
                            f.write(f"  ⚠️ WARNING: Rank is too low (>{rank})\n")
                        if imp < 0.01:
                            f.write(f"  ⚠️ WARNING: Importance is too low (<{imp:.6f})\n")
                    else:
                        f.write(f"\n❌ is_weekend NOT FOUND\n")
                    
                    festivo_row = df[df['feature'] == 'is_festivo']
                    if not festivo_row.empty:
                        rank = df.index.get_loc(festivo_row.index[0]) + 1
                        imp = festivo_row.iloc[0]['importance']
                        f.write(f"\nis_festivo:\n")
                        f.write(f"  Rank: #{rank} / {len(df)}\n")
                        f.write(f"  Importance: {imp:.6f}\n")
                    else:
                        f.write(f"\n❌ is_festivo NOT FOUND\n")
                    
                    # Save full analysis
                    df.to_csv('feature_importance_full.csv', index=False)
                    f.write(f"\nFull analysis saved to: feature_importance_full.csv\n")
    except Exception as e:
        f.write(f"\nERROR: {str(e)}\n")
        import traceback
        f.write(traceback.format_exc())

print("Analysis written to model_analysis_output.txt")

