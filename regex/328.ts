import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*username)(?=.*user name)((?=.*refuse)|(?=.*reject)|(?=.*remove)|(?=.*disable)|(?=.*cancel)))", "i"),
	caseID: 328
} as Regex;