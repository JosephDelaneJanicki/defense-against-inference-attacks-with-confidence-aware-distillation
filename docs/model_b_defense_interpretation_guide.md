# Model B Defense Interpretation Guide

## Overview

This guide explains the purpose and behavior of each defense applied
when training Model B (student models).

## Core Idea

All defenses aim to preserve useful patterns while removing memorization
signals.

## Defenses

### Raw Soft Labels

Uses full probability outputs. High utility, high leakage.

### Hard Labels

Uses argmax labels. Medium utility, reduced leakage.

### Temperature Smoothing

Softens probabilities. Good utility, reduced leakage.

### Confidence Clipping

Caps extreme probabilities. Slight utility loss, reduced leakage.

### Confidence Filtering

Removes high-confidence samples. Lower utility, strong privacy.

### Feature Anonymization

Generalizes inputs. Moderate utility loss, improved privacy.

### Combined Defenses

Applies multiple methods. Best tradeoff.

## Naming Convention

student\_`<label_processing>`{=html}\_`<feature_processing>`{=html}

## Summary

Utility = performance\
Privacy = resistance to attack\
Best defense balances both.
