import os

import nibabel as nib


class TumorSaver:
    @staticmethod
    def get_datatype(datatype):
        datatype_map = {
            2: 'uint8',
            4: 'int16',
            8: 'int32',
            16: 'float32',
            32: 'complex64',
            64: 'float64'
        }
        return datatype_map.get(datatype.item(), 'uint8')

    @staticmethod
    def save_nifti(data, affine_matrix, data_type, output_path, filename):
        os.makedirs(output_path, exist_ok=True)

        filename = 'synt_' + filename
        full_path = os.path.join(output_path, filename)

        if os.path.exists(full_path):
            counter = 1
            filename, _ = os.path.splitext(filename)
            filename, _ = os.path.splitext(filename)
            final_filename = f"{filename}_{counter}"
            final_filename += '.nii.gz'
            while os.path.exists(os.path.join(output_path, final_filename)):
                counter += 1
                final_filename = f"{filename}_{counter}"
                final_filename += '.nii.gz'

            filename = f"{filename}_{counter}"
            filename += '.nii.gz'
            full_path = os.path.join(output_path, filename)

        nib.save(
            nib.Nifti1Image(data.astype(data_type), affine_matrix),
            full_path
        )
        print(f"Saved {full_path}")

    @staticmethod
    def save_data(d, folder='default'):
        image_data_type = TumorSaver.get_datatype(d['image_meta_dict']['datatype'])
        image_affine_matrix = d['image_meta_dict']['original_affine'][0]

        label_data_type = TumorSaver.get_datatype(d['label_meta_dict']['datatype'])
        label_affine_matrix = d['label_meta_dict']['original_affine'][0]

        image = d['image'][0].squeeze(0).cpu().numpy()
        label = d['label'][0].squeeze(0).cpu().numpy()

        image_outputs = os.path.join('synt', folder, 'image')
        label_outputs = os.path.join('synt', folder, 'label')

        image_filename = os.path.basename(d['image_meta_dict']['filename_or_obj'][0])
        label_filename = os.path.basename(d['label_meta_dict']['filename_or_obj'][0])

        TumorSaver.save_nifti(image, image_affine_matrix, image_data_type, image_outputs, image_filename)
        TumorSaver.save_nifti(label, label_affine_matrix, label_data_type, label_outputs, label_filename)

