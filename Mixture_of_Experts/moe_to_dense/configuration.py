"""Validate the synthetic experiment configuration at the entry points."""
import math


def validate_config(config: dict) -> dict:
    def integer(value, name, minimum=1):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    def positive(value, name):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    try:
        integer(config['seed'], 'seed', 0)
        integer(config.get('torch_threads', 1), 'torch_threads')
        integer(config['benchmark_repeats'], 'benchmark_repeats')
        model, data, conversion = (config[key] for key in ('model', 'data', 'conversion'))
        for key in ('vocab_size', 'hidden_size', 'intermediate_size', 'num_experts', 'top_k'):
            integer(model[key], key)
        for key in ('train_samples', 'calibration_samples', 'evaluation_samples', 'batch_size'):
            integer(data[key], key)
        integer(data['sequence_length'], 'sequence_length', 2)
        for key in ('selected_count', 'group_count'):
            integer(conversion[key], key)
        if not conversion['group_count'] <= conversion['selected_count'] <= model['num_experts'] or model['top_k'] > model['num_experts']:
            raise ValueError('expert counts are inconsistent')
        if conversion['scaling'] not in ('uniform', 'proportional'):
            raise ValueError('unknown scaling')
        if conversion['lambda_reg'] is not None:
            positive(conversion['lambda_reg'], 'lambda_reg')
        methods = config['methods']
        if not isinstance(methods, list) or not methods or any(m not in ('sf', 'cp', 'acp', 'do_cp', 'do_acp') for m in methods) or len(set(methods)) != len(methods):
            raise ValueError('methods must be unique supported scoring methods')
        for section in ('teacher_training', 'distillation'):
            integer(config[section]['steps'], f'{section}.steps')
            positive(config[section]['learning_rate'], f'{section}.learning_rate')
        kd = config['distillation']
        integer(kd['gradient_accumulation_steps'], 'gradient_accumulation_steps')
        integer(kd['checkpoint_every'], 'checkpoint_every', 0)
        positive(kd['temperature'], 'temperature')
    except (KeyError, TypeError) as exc:
        raise ValueError(f'invalid or incomplete configuration: {exc}') from exc
    return config
