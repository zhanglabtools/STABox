#!/usr/bin/env python3
import os
import sys
import shutil
import json
import numpy as np
from .h5ad_wrapper import H5ADWrapper
from .obj_wrapper import OBJWrapper
from .webserver import webserver_main



def check_file(filename):
    if not os.path.isfile(filename):
        print(f'Error: invalid input file :{filename}!', flush=True)
        sys.exit(3)


def create_folder(foldername):
    try:
        os.mkdir(foldername)
        print(f'create {foldername} ....')
    except FileExistsError:
        print(f'cache folder -- {foldername} already exists, reuse it now....')


class int64_encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def savedata2json(data, filename):
    text = json.dumps(data, cls=int64_encoder)
    textfile = open(filename, "w")
    textfile.write(text)
    textfile.close()


#####################################################
# Usage
#
def webcache_usage():
    print("""
Usage : vt3d AtlasBrowser BuildAtlas [options]

Options:
       required options:
            -i <input.h5ad>
            -c <conf.json>
            -o [output prefix, default webcace]
Example:
        > vt3d AtlasBrowser BuildAtlas -i in.h5ad -c atlas.json
        > cat atlas.json
        {
            "Coordinate" : "spatial3D",
            "Annotatinos" : [ "lineage" ],
            "Meshes" : {
                "body" : "example_data/body.obj" ,
                "gut" : "example_data/gut.obj"     ,
                "nueral" : "example_data/neural.obj" ,
                "pharynx" : "example_data/pharynx.obj"
            },
            "mesh_coord" : "example_data/WT.coord.json",
            "Genes" : [
               "SMED30033583" ,
               "SMED30011277" ,
       ... genes you want to display ...
               "SMED30031463" ,
               "SMED30033839"
            ]
        }

Notice:
     Set "Genes" : ["all"] to export all genes in var.

The structure of output atlas folder:
      webcache
        +---Anno
             +---lineage.json
        +---Gene
             +---SMED30033583.json
             +---SMED30011277.json
             ...
             +---SMED30031463.json
             +---SMED30033839.json
        +---summary.json
        +---gene.json
        +---meshes.json
""", flush=True)


#####################################################
# main pipe
#
def webcache_main(filepath, inh5data_, conf_file_, data_name):
    prefix = 'webcache'
    prefix, inh5data, conf_file = os.path.join(filepath, prefix), os.path.join(filepath, inh5data_), os.path.join(
        filepath, conf_file_)

    if inh5data == '' or prefix == '' or conf_file == '':
        print('Error: incomplete parameters, exit ...')
        webcache_usage()
        sys.exit(2)

    confdata = json.load(open(conf_file))
    if (not 'Coordinate' in confdata) \
            or (not 'Annotatinos' in confdata) \
            or (not 'Meshes' in confdata) \
            or (not 'Genes' in confdata):
        print('Error: incomplete conf.json, please copy from usage!', flush=True)
        sys.exit(3)
    if len(confdata['Annotatinos']) < 1 and len(confdata['Genes']) < 1 and len(confdata['Meshes']) < 1:
        print('Error: nothing to show! exit ...', flush=True)
        sys.exit(4)
    for meshkey in confdata['Meshes']:
        if not 'mesh_coord' in confdata:
            print('Error: no mesh_coord exit ...', flush=True)
            sys.exit(4)
        meshfile = os.path.join(filepath, confdata['Meshes'][meshkey])
        check_file(os.path.join(filepath, confdata['mesh_coord']))
        check_file(meshfile)

    #######################################
    # load h5ad and sanity check
    inh5ad = H5ADWrapper(inh5data)
    if data_name == 'Single-DLPFC':
        inh5ad.data.obsm['spatial'] = np.c_[inh5ad.data.obsm['spatial'], np.array([1000]*inh5ad.data.obsm['spatial'].shape[0])]
        for annokey in confdata['Annotatinos']:
            if not inh5ad.hasAnno(annokey):
                print(f'Error: invalid Annotatino :{annokey}!', flush=True)
                sys.exit(3)
        if "all" in confdata['Genes']:
            confdata['Genes'] = inh5ad.AllGenesList()
        else:
            for genename in confdata['Genes']:
                if not inh5ad.hasGene(genename):
                    print(f'Error: invalid gene :{genename}!', flush=True)
                    sys.exit(3)
        if not inh5ad.hasCoord(confdata['Coordinate']):
            print(f'Error: invalid Coordinate :{confdata["Coordinate"]}!', flush=True)
            sys.exit(3)

        # create main folder, summary.json gene.json
        create_folder(f'{prefix}')
        create_folder(f'{prefix}/Anno')
        create_folder(f'{prefix}/Gene')

        # generate mesh json
        if len(confdata['Meshes']) > 1:
            coord_file = os.path.join(filepath, confdata['mesh_coord'])
            meshes = OBJWrapper(coord_file)
            for meshname in confdata['Meshes']:
                meshes.add_mesh(meshname, os.path.join(filepath, confdata['Meshes'][meshname]))
            savedata2json(meshes.get_data(), f'{prefix}/meshes.json')

        # generate summary and gene json
        summary = inh5ad.getSummary(confdata['Coordinate'], confdata['Annotatinos'], confdata['Genes'])
        if len(confdata['Meshes']) > 1:
            summary = meshes.update_summary(summary)
        summary['annomapper']['mclust_legend2int'] = dict([(str(x), y) for x, y in summary['annomapper']['mclust_legend2int'].items()])
        summary['annomapper']['mclust_int2legend'] = dict([(str(x), y) for x, y in summary['annomapper']['mclust_int2legend'].items()])

        savedata2json(summary, f'{prefix}/summary.json')
        savedata2json(confdata['Genes'], f'{prefix}/gene.json')

        # generate annotation json
        for anno in confdata['Annotatinos']:
            xyza = inh5ad.getCellXYZA(confdata['Coordinate'], int, anno)
            mapper = summary['annomapper'][f'{anno}_legend2int']
            xyza['annoid'] = xyza.apply(lambda row: mapper[str(row['anno'])], axis=1)
            savedata2json(xyza[['x', 'y', 'z', 'annoid']].to_numpy().tolist(), f'{prefix}/Anno/{anno}.json')

        # generate gene json
        for gene in confdata['Genes']:
            s = '"'
            ss = ':'
            if s in gene:
                new_gene = gene.replace('"', '')
            else:
                new_gene = gene
            if ss in gene:
                new_gene = gene.replace(':', '-')
            xyze = inh5ad.getGeneXYZE(gene, 0, confdata['Coordinate'], int)
            savedata2json(xyze.to_numpy().tolist(), f'{prefix}/Gene/{new_gene}.json')

        # cp html and js
        base = '../Data_Browser'
        flist = ['6f0a76321d30f3c8120915e57f7bd77e.ttf',
                 'index.html',
                 'index.js',
                 'index.js.map',
                 'manifest.js',
                 'manifest.js.map',
                 'vendor.js',
                 'vendor.js.map', ]
        for xx in flist:
            shutil.copyfile(f'{base}/{xx}', f'{prefix}/{xx}')

        os.chdir('../H5AD_Save/DLPFC/webcache')
        webserver_main(ports=8050)
    elif data_name == 'Multi-DLPFC':
        section_id = list(set(inh5ad.data.obs['batch_name'].values))
        spot_Z = []
        for i in range(len(section_id)):
            id_lens = len(inh5ad.data.obs[inh5ad.data.obs['batch_name'] == section_id[i]].index)
            temp_list = [1000 * (i + 1)] * id_lens
            spot_Z.extend(temp_list)

        inh5ad.data.obsm['spatial'] = np.c_[inh5ad.data.obsm['spatial'], np.array(spot_Z)]
        for annokey in confdata['Annotatinos']:
            if not inh5ad.hasAnno(annokey):
                print(f'Error: invalid Annotatino :{annokey}!', flush=True)
                # sys.exit(3)
        if "all" in confdata['Genes']:
            confdata['Genes'] = inh5ad.AllGenesList()
        else:
            for genename in confdata['Genes']:
                if not inh5ad.hasGene(genename):
                    print(f'Error: invalid gene :{genename}!', flush=True)
                    sys.exit(3)
        if not inh5ad.hasCoord(confdata['Coordinate']):
            print(f'Error: invalid Coordinate :{confdata["Coordinate"]}!', flush=True)
            sys.exit(3)

        # create main folder, summary.json gene.json
        create_folder(f'{prefix}')
        create_folder(f'{prefix}/Anno')
        create_folder(f'{prefix}/Gene')

        # generate mesh json
        if len(confdata['Meshes']) > 1:
            coord_file = os.path.join(filepath, confdata['mesh_coord'])
            meshes = OBJWrapper(coord_file)
            for meshname in confdata['Meshes']:
                meshes.add_mesh(meshname, os.path.join(filepath, confdata['Meshes'][meshname]))
            savedata2json(meshes.get_data(), f'{prefix}/meshes.json')

        # generate summary and gene json
        summary = inh5ad.getSummary(confdata['Coordinate'], confdata['Annotatinos'], confdata['Genes'])
        if len(confdata['Meshes']) > 1:
            summary = meshes.update_summary(summary)
        # print(summary,flush=True)
        summary['annomapper']['mclust_legend2int'] = dict(
            [(str(x), y) for x, y in summary['annomapper']['mclust_legend2int'].items()])
        summary['annomapper']['mclust_int2legend'] = dict(
            [(str(x), y) for x, y in summary['annomapper']['mclust_int2legend'].items()])

        savedata2json(summary, f'{prefix}/summary.json')
        savedata2json(confdata['Genes'], f'{prefix}/gene.json')
        #######################################
        # generate annotation json
        for anno in confdata['Annotatinos']:
            xyza = inh5ad.getCellXYZA(confdata['Coordinate'], int, anno)
            mapper = summary['annomapper'][f'{anno}_legend2int']
            xyza['annoid'] = xyza.apply(lambda row: mapper[str(row['anno'])], axis=1)
            savedata2json(xyza[['x', 'y', 'z', 'annoid']].to_numpy().tolist(), f'{prefix}/Anno/{anno}.json')
        #######################################
        # generate gene json
        for gene in confdata['Genes']:
            s = '"'
            ss = ':'
            if s in gene:
                new_gene = gene.replace('"', '')
            else:
                new_gene = gene
            if ss in gene:
                new_gene = gene.replace(':', '-')
            xyze = inh5ad.getGeneXYZE(gene, 0, confdata['Coordinate'], int)
            savedata2json(xyze.to_numpy().tolist(), f'{prefix}/Gene/{new_gene}.json')
        #######################################
        # cp html and js
        base = '../Data_Browser'
        flist = ['6f0a76321d30f3c8120915e57f7bd77e.ttf',
                 'index.html',
                 'index.js',
                 'index.js.map',
                 'manifest.js',
                 'manifest.js.map',
                 'vendor.js',
                 'vendor.js.map', ]
        for xx in flist:
            shutil.copyfile(f'{base}/{xx}', f'{prefix}/{xx}')
        os.chdir('../H5AD_Save/STAligner_Multi-DLPFC/webcache')
        webserver_main(ports=8050)
    elif data_name == '3D_HippoData':
        inh5ad.data.obsm['spatial3D'] = np.c_[inh5ad.data.obs['X'], inh5ad.data.obs['Y'], inh5ad.data.obs['Z']]
        for annokey in confdata['Annotatinos']:
            if not inh5ad.hasAnno(annokey):
                print(f'Error: invalid Annotatino :{annokey}!', flush=True)
                sys.exit(3)
        if "all" in confdata['Genes']:
            confdata['Genes'] = inh5ad.AllGenesList()
        else:
            for genename in confdata['Genes']:
                if not inh5ad.hasGene(genename):
                    print(f'Error: invalid gene :{genename}!', flush=True)
                    sys.exit(3)
        if not inh5ad.hasCoord(confdata['Coordinate']):
            print(f'Error: invalid Coordinate :{confdata["Coordinate"]}!', flush=True)
            sys.exit(3)

        # create main folder, summary.json gene.json
        create_folder(f'{prefix}')
        create_folder(f'{prefix}/Anno')
        create_folder(f'{prefix}/Gene')
        #######################################
        # generate mesh json
        if len(confdata['Meshes']) > 1:
            coord_file = os.path.join(filepath, confdata['mesh_coord'])
            meshes = OBJWrapper(coord_file)
            for meshname in confdata['Meshes']:
                meshes.add_mesh(meshname, os.path.join(filepath, confdata['Meshes'][meshname]))
            savedata2json(meshes.get_data(), f'{prefix}/meshes.json')
        #######################################
        # generate summary and gene json
        summary = inh5ad.getSummary(confdata['Coordinate'], confdata['Annotatinos'], confdata['Genes'])
        if len(confdata['Meshes']) > 1:
            summary = meshes.update_summary(summary)
        summary['annomapper']['mclust_legend2int'] = dict(
            [(str(x), y) for x, y in summary['annomapper']['mclust_legend2int'].items()])
        summary['annomapper']['mclust_int2legend'] = dict(
            [(str(x), y) for x, y in summary['annomapper']['mclust_int2legend'].items()])

        savedata2json(summary, f'{prefix}/summary.json')
        savedata2json(confdata['Genes'], f'{prefix}/gene.json')
        #######################################
        # generate annotation json
        for anno in confdata['Annotatinos']:
            xyza = inh5ad.getCellXYZA(confdata['Coordinate'], int, anno)
            mapper = summary['annomapper'][f'{anno}_legend2int']
            xyza['annoid'] = xyza.apply(lambda row: mapper[str(row['anno'])], axis=1)
            savedata2json(xyza[['x', 'y', 'z', 'annoid']].to_numpy().tolist(), f'{prefix}/Anno/{anno}.json')
        #######################################
        # generate gene json
        for gene in confdata['Genes']:
            s = '"'
            ss = ':'
            if s in gene:
                new_gene = gene.replace('"', '')
            else:
                new_gene = gene
            if ss in gene:
                new_gene = gene.replace(':', '-')
            xyze = inh5ad.getGeneXYZE(gene, 0, confdata['Coordinate'], int)
            savedata2json(xyze.to_numpy().tolist(), f'{prefix}/Gene/{new_gene}.json')
        #######################################
        # cp html and js
        base = os.path.dirname(os.path.realpath(__file__))
        base = f'{base}/../vt3d_browser'
        flist = ['6f0a76321d30f3c8120915e57f7bd77e.ttf',
                 'index.html',
                 'index.js',
                 'index.js.map',
                 'manifest.js',
                 'manifest.js.map',
                 'vendor.js',
                 'vendor.js.map', ]
        for xx in flist:
            shutil.copyfile(f'{base}/{xx}', f'{prefix}/{xx}')
        os.chdir('../H5AD_Save/STAGATE_3DSpatialData/webcache')
        webserver_main(ports=8050)


def webServer(path, ports):
    # os.chdir('../H5AD_Save/DLPFC/webcache')
    # webserver_main(ports=8050)
    print(f'path is {path}')
    print(f'current running path is {os.getcwd()}')
    running_path = os.getcwd()
    # os.chdir(os.getcwd() + '/DLPFC/webcache')
    os.chdir(os.getcwd() + path)
    print(f'current running path is {os.getcwd()}')
    webserver_main(ports=ports)
    os.chdir(running_path)

# if __name__ == '__main__':
    # run Stereo-seq-L1
    # filepath = 'D:/Users/lqlu/download/3D_SRT_Data/Stereo-seq-L1'
    # inh5data = 'L1_a_count_normal_stereoseq.h5ad'
    # conf_file = 'L1_a.atlas.json'
    # webcache_main(filepath, inh5data, conf_file)
    # run STARMap
    # filepath = 'D:/Users/lqlu/download/3D_SRT_Data/DLPFC'
    # inh5data = '10X_151507_filtered_feature_bc_matrix.h5ad'
    # conf_file = '10X_DLPFC.atlas.json'
    # webcache_main(filepath, inh5data, conf_file)



