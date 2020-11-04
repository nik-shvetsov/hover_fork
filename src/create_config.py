import argparse
import yaml

info = {
'nuclei_types_hv_consep': {
    'Misc': 1,
    'Inflammatory': 2,
    'Epithelial': 3,
    'Spindle': 4
},
'nuclei_types_hv_pannuke': {
    'Inflammatory': 1,
    'Connective': 2,
    'Dead cells': 3,
    'Epithelial': 4,
    'Neoplastic cells': 5
}}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True, choices=['hv_consep', 'hv_pannuke'])
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_type', required=True, choices=['np_hv', 'np_hv_opt'])
    parser.add_argument('--preproc', default=False, action='store_true')
    parser.add_argument('--step_size', help='80 if consep, 164 if pannuke', required=True)
    parser.add_argument('--win_size', help='540 if consep, 256 if pannuke', required=True)
    

    parser.add_argument('--inf_auto_find_chkpt', action='store_true')
    parser.add_argument('--inf_model')

    parser.add_argument('--id', default='1.0')
    parser.add_argument('--input_prefix', default='/data/input/')
    parser.add_argument('--output_prefix', default='/data/output/')

    parser.add_argument('--extract', default=True, action='store_true')
    parser.add_argument('--export', default=True, action='store_true')

    parser.add_argument('--extract_type', default='mirror')

    parser.add_argument('--input_norm', default=True, action='store_true')
    parser.add_argument('--input_augs', default='p_standard')
    parser.add_argument('--train_dir', default='train/Annotations')
    parser.add_argument('--valid_dir', default='valid/Annotations')

    parser.add_argument('--data_modes', default='train,test')

    parser.add_argument('--inf_data_list', required=True)
    parser.add_argument('--remap_labels', default=True, action='store_true')
    parser.add_argument('--outline', default='')

    parser.add_argument('--norm_brightness', default=True, action='store_true')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--image', default='train_1')

    parser.add_argument('--img_ext', default='.png')
    parser.add_argument('--data_ext', default='.npy')
    parser.add_argument('--inf_imgs_ext', default='.png')
    parser.add_argument('--nr_procs_train', default=8)
    parser.add_argument('--nr_procs_valid', default=4)

    args = parser.parse_args()

    config_dict = {
        f'{args.profile}': # H_PROFILE
        {   
            'mode': 'hover',
            'model_type': args.model_type,
            'seed': 10,
            'type_classification': True,

            'exp_id': args.id,
            'input_prefix': args.input_prefix,
            'output_prefix': args.output_prefix,
            'include_preproc': args.preproc,
            'include_extract': args.extract,
            'include_export': args.export,
            
            'stain_norm':
            {
                'norm_brightness': args.norm_brightness,
                'mode': args.mode,
                'image': args.image
            } if args.preproc else None,

            'data_dir': args.data_dir,
            'extract_type': args.extract_type,
            # 'img_ext': args.img_ext,
            'data_modes': args.data_modes.split(','),
            'step_size': [int(args.step_size), int(args.step_size)],
            'win_size': [int(args.win_size), int(args.win_size)],
            'nuclei_types': info[f'nuclei_types_{args.profile}'],
            # 'data_ext': args.data_ext,
            'input_norm': args.input_norm,
            'input_augs': args.input_augs,
            'train_dir': [ args.train_dir ],
            'valid_dir': [ args.valid_dir ],
            # 'nr_procs_train': args.nr_procs_train,
            # 'nr_procs_valid': args.nr_procs_valid,
  
            'inf_auto_find_chkpt' if args.inf_auto_find_chkpt else 'inf_model': \
                args.inf_auto_find_chkpt if args.inf_auto_find_chkpt else args.inf_model,

            # 'inf_imgs_ext': args.inf_imgs_ext,
            'inf_data_list': [ args.inf_data_list ],
            'remap_labels': args.remap_labels,
            'outline': args.outline
        }
    }

    with open('config.yml', 'w') as file:
        yaml.dump(config_dict, file)