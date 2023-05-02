from api.utils_scanssd.convert_pdf_to_image import create_images_from_pdfs
import streamlit as st
import os
import base64
import glob
import uuid
import openai
from pix2text import Pix2Text

path_list = ['images','pdf','results','annotation']
root_folder = '../pix2text'
uuid_folder = None
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

#criando estrutura mínima de diretórios
@st.cache_data
def create_folders(root_folder,path_list):
    uuid_folder = (str(uuid.uuid4())).split('-',1)[0]
    #uuid_folder = 'test'
    if not os.path.exists(os.path.join(root_folder,uuid_folder)):
        os.mkdir(os.path.join(root_folder,uuid_folder))
    paths = [os.path.join(root_folder,uuid_folder,path) for path in path_list]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
    return uuid_folder

def create_latex_file(root_folder,exp_name,results):
    with open(os.path.join(root_folder,exp_name,'latex.txt'),'w') as f:
        f.write(results)
        f.close()

def predict(
    img_file_or_dir,
    save_analysis_res,
    use_analyzer=True,
    analyzer_name='mfd',
    analyzer_type='yolov7_tiny',
    device='cpu',
    resized_shape=600):

    p2t = Pix2Text(
        analyzer_config=dict(model_name=analyzer_name, model_type=analyzer_type),
        device=device,
    )

    fp_list = []
    if os.path.isfile(img_file_or_dir):
        fp_list.append(img_file_or_dir)
        if save_analysis_res:
            save_analysis_res = [save_analysis_res]
    elif os.path.isdir(img_file_or_dir):
        fn_list = glob.glob1(img_file_or_dir, '*g')
        fp_list = [os.path.join(img_file_or_dir, fn) for fn in fn_list]
        if save_analysis_res:
            os.makedirs(save_analysis_res, exist_ok=True)
            save_analysis_res = [
                os.path.join(save_analysis_res, 'analysis-' + fn) for fn in fn_list
            ]
    res = ""
    for idx, fp in enumerate(fp_list):
        analysis_res = save_analysis_res[idx] if save_analysis_res is not None else None
        print(analysis_res)
        print(fp)
        out = p2t.recognize(fp,use_analyzer=use_analyzer,resized_shape=resized_shape,save_analysis_res=analysis_res)
        res = res.join('\n'.join([o['text'] for o in out if o['type']=='isolated']))

    create_latex_file(root_folder,uuid_folder,res)

uuid_folder = create_folders(root_folder=root_folder,path_list=path_list)

file = st.file_uploader("Please choose a file", type="pdf")

if file is not None:
    # file_details = {"FileName": file.name, "FileType": file.type}
    # st.write(file_details)
    st.error(f"Do you really want to process the file: {file.name}")
    if st.button("Yes"):
        st.session_state["process"] = True
        file_path = os.path.join(root_folder,uuid_folder,'pdf', file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        show_pdf(file_path)

        st.success("File Uploaded")
        with st.spinner('Gerando Imagens a partir do PDF...'):
            file_name = create_images_from_pdfs(root_folder,uuid_folder)
            
        annot_images = os.path.join(root_folder,uuid_folder,'annotation')
        st.success("Imagens Geradas!")

        # read image from response
        file_name = (file.name).split('.',1)[0]
        #print(os.path.join(annot_images,file_name,'*.png'))
        st.write('images read')

        # Processamento com Pix2Text
        dir_images = os.path.join(root_folder,uuid_folder,'images',file_name)
        results = os.path.join(root_folder,uuid_folder,'annotation')
        with st.spinner('Anotando Imagens...'):
            predict(dir_images,results)

        st.success("Imagens Anotadas!")

        images_to_plot = glob.glob(os.path.join(annot_images,'*.png'))
        # plot images in 3 columns
        grid_size = 3
        image_groups = []
        for i in range(0, len(images_to_plot), grid_size):
            image_groups.append(images_to_plot[i:i+grid_size])

        for image_group in image_groups:
            streamlit_columns = st.columns(grid_size)
            for i, image in enumerate(image_group):
                streamlit_columns[i].image(image)


        # API Setup
        API_KEY = "841943d7281d469ba9233ba8aba57755"
        RESOURCE_ENDPOINT = "https://alanai.openai.azure.com/"

        # pre-set
        openai.api_key = API_KEY
        openai.api_base = RESOURCE_ENDPOINT
        openai.api_type = 'azure'
        openai.api_version = '2022-12-01'  # this may change in the future

        # Models
        deployment_name = "alan-gpt-35-turbo0301"

        ## Aqui pode exibir o latex gerado a partir do arquivo
        ## root_folder/uuid_folder/latex.txt
        ## depois para cada linha desse arquivo, fazer uma chamada
        ## para a API do chatGPT e exibir o resultado
        
        with st.spinner('Aguarde o retorno do GPT...'):
            with open(os.path.join(root_folder,uuid_folder, 'latex.txt')) as file:
                for line in file:
                    if line != "" and '$$' not in line:
                        print(line)
                        start_phrase = f'Identify and Describe the equation: {line}'
                        ###
                        response = openai.Completion.create(engine=deployment_name, prompt=start_phrase, max_tokens=500)
                        text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
                        st.write(f"Equation: {line}, result: {text}")
        st.success("Sucesso")


