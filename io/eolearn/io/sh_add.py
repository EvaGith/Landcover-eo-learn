"""
Module for adding data obtained from sentinelhub package to existing EOPatches
"""

import numpy as np
import logging

from sentinelhub import WmsRequest, WcsRequest, MimeType, DataSource, \
    CustomUrlParam, ServiceType

from eolearn.core import EOTask, FeatureType, get_common_timestamps

LOGGER = logging.getLogger(__name__)


class AddSentinelHubOGCFeature(EOTask):
    """
    Task for adding feature to existing EOPatch using Sentinel Hub's OGC request. The following OGC request
    parameters are taken from the EOPatch's meta data (set by the request that created the EOPatch):
    * size_x: width or resx
    * size_y: height or resy
    * maxcc: max cloud coverage
    * time_difference
    * service_type: WMS or WCS
    * time_interval

    In case the available dates in feature request don't match the dates of frames in the EOPatch the
    timestamp consolidation is triggered. This means that if for example and EOPatch has data for the following dates
    (noted as integers for brevity):
        * eopatch.timestamps = [1, 2, 3, 4, 5],
    while available dates for the new feature are:
        * new_feature_request.get_dates() = [1, 3, 4, 5, 6]
        * note date 2 is missing and 6 is additional in the request,
    then this task will
        * download new feature data only for the dates [1,3,4,5]
        * and will remove all data for date [2] in the eopatch.
    Finally, the eopatch will at the end contain data for dates [1, 3, 4, 5] for old and new features.

    :param layer: the preconfigured layer to be added to EOPatch. Required.
    :type layer: str
    :param feature_name: user specified name (key) for this feature. Optional.
    :type feature_name: str or None. Default is the same as layer.
    :param data_source: Source of requested satellite data.
    :type data_source: constants.DataSource
    :param image_format: format of the returned image by the Sentinel Hub's WMS getMap service. Default is 32-bit TIFF.
    :type image_format: constants.MimeType
    :param instance_id: user's instance id. If ``None`` the instance id is taken from the ``config.json``
                        configuration file from sentinelhub-py package.
    :type instance_id: str or None
    :param custom_url_params: dictionary of CustomUrlParameters and their values supported by Sentinel Hub's WMS and WCS
                              services. All available parameters are described at
                              http://www.sentinel-hub.com/develop/documentation/api/custom-url-parameters. Note: in
                              case of constants.CustomUrlParam.EVALSCRIPT the dictionary value must be a string
                              of Javascript code that is not encoded into base64.
    :type custom_url_params: dictionary of CustomUrlParameter enum and its value, i.e.
                              ``{constants.CustomUrlParam.ATMFILTER:'ATMCOR'}``
    """

    def __init__(self, feature_type, layer, feature_name=None, data_source=None,
                 image_format=MimeType.TIFF_d32f, instance_id=None, custom_url_params=None):

        self.feature_type = feature_type
        self.layer = layer
        self.feature_name = layer if feature_name is None else feature_name
        self.data_source = data_source
        self.image_format = image_format
        self.instance_id = instance_id

        custom_params = {CustomUrlParam.SHOWLOGO: False,
                         CustomUrlParam.TRANSPARENT: False}
        if custom_url_params is None:
            self.custom_url_params = custom_params
        else:
            self.custom_url_params = {**custom_params, **custom_url_params}

    def _get_wms_request(self, bbox, time_interval, size_x, size_y, maxcc, time_difference):
        """
        Returns WMS request.
        """
        return WmsRequest(layer=self.layer,
                          bbox=bbox,
                          time=time_interval,
                          width=size_x,
                          height=size_y,
                          maxcc=maxcc,
                          image_format=self.image_format,
                          custom_url_params=self.custom_url_params,
                          time_difference=time_difference,
                          data_source=self.data_source,
                          instance_id=self.instance_id)

    def _get_wcs_request(self, bbox, time_interval, size_x, size_y, maxcc, time_difference):
        """
        Returns WCS request.
        """
        return WcsRequest(layer=self.layer,
                          bbox=bbox,
                          time=time_interval,
                          resx=size_x, resy=size_y,
                          maxcc=maxcc,
                          image_format=self.image_format,
                          custom_url_params=self.custom_url_params,
                          time_difference=time_difference,
                          data_source=self.data_source,
                          instance_id=self.instance_id)

    def _reshape_array(self, array, dims_dict):
        """ Reshape array if dimensions do not match requirements

        :param array: Input array
        :param dims_dict: Dictionary with target dimensionality for the feature types
        :return: Reshaped array with additional channel
        """
        if array.ndim == dims_dict[self.feature_type.value] - 1:
            return array.reshape(array.shape + (1,))
        return array

    def _check_dimensionality(self, array, dims_dict):
        """ Method to ensure array has the dimensionality required by the feature type

        :param array: Input array
        :param dims_dict: Dictionary with target dimensionality for the feature types
        :return: Reshaped array with additional channel
        """
        if self.feature_type in [FeatureType.DATA, FeatureType.MASK]:
            return self._reshape_array(array, dims_dict)
        elif self.feature_type in [FeatureType.DATA_TIMELESS, FeatureType.MASK_TIMELESS]:
            array = array.squeeze(axis=0)
            return self._reshape_array(array, dims_dict)
        return array

    def execute(self, eopatch):
        """
        Add requested feature to this existing EOPatch.
        """
        size_x = eopatch.meta_info['size_x']
        size_y = eopatch.meta_info['size_y']
        maxcc = eopatch.meta_info['maxcc']
        time_difference = eopatch.meta_info['time_difference']
        service_type = eopatch.meta_info['service_type']
        time_interval = (eopatch.timestamp[0].isoformat(), eopatch.timestamp[-1].isoformat())

        request = {ServiceType.WMS: self._get_wms_request,
                   ServiceType.WCS: self._get_wcs_request}[service_type](eopatch.bbox, time_interval, size_x, size_y,
                                                                         maxcc, time_difference)

        # check timestamp consistency between request and this eopatch
        request_dates = request.get_dates()
        download_frames = get_common_timestamps(request_dates, eopatch.timestamp)

        request_return = request.get_data(raise_download_errors=False, data_filter=download_frames)
        bad_data = [idx for idx, value in enumerate(request_return) if value is None]
        for idx in reversed(sorted(bad_data)):
            LOGGER.warning('Data from %s could not be downloaded for %s!', str(request_dates[idx]), self.layer)
            del request_return[idx]
            del request_dates[idx]

        request_data = np.asarray(request_return)

        removed_frames = eopatch.consolidate_timestamps(request_dates)
        for rm_frame in removed_frames:
            LOGGER.warning('Removed data for frame %s from eopatch '
                           'due to unavailability of %s!', str(rm_frame), self.layer)

        request_data = self._check_dimensionality(request_data, eopatch.ndims)

        eopatch.add_feature(self.feature_type, self.feature_name, request_data)

        return eopatch


class AddSen2CorClassificationFeature(AddSentinelHubOGCFeature):
    """
    Adds SCL (scene classification), CLD (cloud probability) or SNW (snow probability)
    Sen2Cor classification results to EOPatch's MASK or DATA feature. The feature is added to MASK (SCL) or
    DATA (CLD, SNW) feature types of EOPatch. The feature names are set to be SCL, CLD or SNW.

    Sen2Cor's scene classification (SCL) contains 11 classes with the following values and meanings:
       * 1 - SC_SATURATED_DEFECTIVE
       * 2 - SC_DARK_FEATURE_SHADOW
       * 3 - SC_CLOUD_SHADOW
       * 4 - VEGETATION
       * 5 - NOT-VEGETATED
       * 6 - WATER
       * 7 - SC_CLOUD_LOW_PROBA / UNCLASSIFIED
       * 8 - SC_CLOUD_MEDIUM_PROBA
       * 9 - CLOUD_HIGH_PROBABILITY
       * 10 - THIN_CIRRUS
       * 11 - SNOW
    """
    def __init__(self, sen2cor_classification, layer, **kwargs):
        # definition of possible types and target features
        classification_types = {'SCL': FeatureType.MASK,
                                'CLD': FeatureType.DATA,
                                'SNW': FeatureType.DATA}

        if sen2cor_classification not in classification_types.keys():
            raise ValueError('Unsupported Sen2Cor classification type: {}.'
                             ' Possible types are: {}'.format(sen2cor_classification, classification_types))

        evalscript = 'return ['+sen2cor_classification+'];'

        super(AddSen2CorClassificationFeature, self).__init__(feature_type=classification_types[sen2cor_classification],
                                                              feature_name=sen2cor_classification,
                                                              layer=layer,
                                                              data_source=DataSource.SENTINEL2_L2A,
                                                              custom_url_params={CustomUrlParam.EVALSCRIPT: evalscript},
                                                              **kwargs)


class AddDEMFeature(AddSentinelHubOGCFeature):
    """
    Adds DEM to DATA_TIMELESS EOPatch feature.
    """
    def __init__(self, layer, **kwargs):
        super(AddDEMFeature, self).__init__(feature_type=FeatureType.DATA_TIMELESS, layer=layer,
                                            data_source=DataSource.DEM, **kwargs)


class AddS2L1CFeature(AddSentinelHubOGCFeature):
    """
    Adds Sentinel-2 L1C feature to EOPatch's DATA feature.
    """
    def __init__(self, layer, **kwargs):
        super(AddS2L1CFeature, self).__init__(feature_type=FeatureType.DATA, layer=layer,
                                              data_source=DataSource.SENTINEL2_L1C, **kwargs)


class AddS2L2AFeature(AddSentinelHubOGCFeature):
    """
    Adds Sentinel-2 L2A feature to EOPatch's DATA feature.
    """
    def __init__(self, layer, **kwargs):
        super(AddS2L2AFeature, self).__init__(feature_type=FeatureType.DATA, layer=layer,
                                              data_source=DataSource.SENTINEL2_L2A, **kwargs)


class AddL8Feature(AddSentinelHubOGCFeature):
    """
    Adds Landsat 8 feature to EOPatch's DATA feature.
    """
    def __init__(self, layer, **kwargs):
        super(AddL8Feature, self).__init__(feature_type=FeatureType.DATA, layer=layer,
                                           data_source=DataSource.LANDSAT8, **kwargs)
