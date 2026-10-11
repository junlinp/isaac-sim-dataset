# Asset provenance

Scene provider: NVIDIA, Isaac Sim 5.1 Office.

- [NVIDIA environment catalog](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_environments.html)
- [Office entry USD](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Office/office.usd)

The generator follows the Office dependency closure, including USD references, shader asset attributes, texture paths, and relative MDL module imports. The tested cache contained 905 files, approximately 713 MB. Original meshes, textures, and MDL source files are downloaded directly from NVIDIA and are not committed here.

The Office is an authored simulation environment. The generator preserves its geometry/materials and adds fixed downward lights at existing ceiling fixtures and fixed corridor fill lights; original illumination leaves several corridors too dark for this capture.

The MIT license in this repository covers the generator code. It does not license NVIDIA content, Isaac Sim, TinyNav, or other third-party dependencies. Review the provider's applicable asset and software terms before downloading, using, or redistributing their content. Generated dataset archives contain cached NVIDIA scene assets, so do not assume the repository's code license applies to those archives.
